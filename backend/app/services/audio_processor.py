"""Обработчик задачи в воркере: скачать исходник → пайплайн → результаты в
S3 + Document → progress.complete. Любая ошибка → progress.error."""

import json
import logging
import shutil
import tempfile
from pathlib import Path

from redis.asyncio import Redis

from app.core.config import settings
from app.core.database import async_session
from app.core.fixtures import REBUILD_STAGES, STAGES_BY_OPERATION, OperationEnum
from app.core.s3 import (
    download_to_path,
    folder_of,
    generate_download_url,
    get_file_from_s3,
    upload_to_s3,
)
from app.core.stored_file import StoredFile
from app.crud.document import create_document, get_document
from app.crud.service import get_service
from app.models.document import Document
from app.schemas.document import DocumentResponse
from app.schemas.progress import Stage
from app.schemas.task_context import AudioTaskContext
from app.services.audio_processing.analysis.stats import dominant_language
from app.services.audio_processing.models import ModelRegistry
from app.services.audio_processing.run import PipelineRunner
from app.services.progress import Progress

logger = logging.getLogger(__name__)


async def _store(
    data: bytes,
    filename: str,
    content_type: str,
    kind: str,
    context: AudioTaskContext,
    language_code: str | None = None,
    usage: dict | None = None,
) -> DocumentResponse:
    if not data:
        raise RuntimeError(f"{kind} result is empty")
    # Результаты лежат рядом с исходником, но в подпапке задачи: перестроение
    # отчёта не должно упираться в уникальность object_key.
    source_folder = folder_of(context.source_object_key) if context.source_object_key else None
    folder = f"{source_folder}/{context.task_id[:8]}" if source_folder else context.task_id[:8]
    stored = StoredFile(filename=filename, data=data, content_type=content_type)
    object_key = await upload_to_s3(stored, folder=folder)
    async with async_session() as db:
        doc: Document = await create_document(
            db=db,
            name=filename,
            object_key=object_key,
            kind=kind,
            language_code=language_code,
            service_name=context.params.model,
            operation_name=context.operation.value,
            parent_id=context.source_document_id,
            usage=usage,
        )
    resp = DocumentResponse.model_validate(doc)
    return resp.model_copy(update={"url": await generate_download_url(resp.object_key)})


def _base_name(context: AudioTaskContext) -> str:
    name = context.source_filename or "audio"
    return name.rsplit(".", 1)[0] if "." in name else name


async def process_audio(
    redis: Redis, models: ModelRegistry, context: AudioTaskContext, worker_id: str | None = None
) -> None:
    progress = Progress(redis=redis, task_id=context.task_id)
    rebuild = context.transcript_document_id is not None
    stages = REBUILD_STAGES if rebuild else STAGES_BY_OPERATION[context.operation]
    tmp_root = Path(settings.audio.tmp_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{context.task_id[:8]}-", dir=tmp_root))

    try:
        await progress.init(total_stages=len(stages), stage=stages[0], worker_id=worker_id)
        llm_service = None
        if context.params.model:
            async with async_session() as db:
                llm_service = await get_service(db, context.params.model)

        runner = PipelineRunner(models=models, progress=progress, work_dir=work_dir, llm_service=llm_service)
        documents: dict[str, dict] = {}
        base = _base_name(context)

        if rebuild:
            async with async_session() as db:
                tdoc = await get_document(db, context.transcript_document_id)
                adoc = await get_document(db, context.analysis_document_id) if context.analysis_document_id else None
            transcript = json.loads((await get_file_from_s3(tdoc.object_key)).read())
            analysis = json.loads((await get_file_from_s3(adoc.object_key)).read()) if adoc else None
            if context.params.speaker_names:
                transcript = runner.apply_speaker_names(transcript, context.params.speaker_names)
            documents["transcript"] = DocumentResponse.model_validate(tdoc).model_copy(
                update={"url": await generate_download_url(tdoc.object_key)}
            ).model_dump(mode="json")
        else:
            input_path = work_dir / f"input.{context.source_filename.rsplit('.', 1)[-1].lower()}"
            await download_to_path(context.source_object_key, str(input_path))
            transcript = await runner.transcribe(
                input_path=input_path,
                filename=context.source_filename,
                job_id=context.task_id,
                params=context.params,
            )
            main_lang = dominant_language(transcript)
            documents["transcript"] = (
                await _store(
                    json.dumps(transcript, ensure_ascii=False, indent=2).encode("utf-8"),
                    f"{base}_transcript.json", "application/json", "transcript", context,
                    language_code=main_lang, usage={"timings_sec": transcript["meta"]["timings_sec"],
                                                    "warnings": transcript["meta"]["warnings"]},
                )
            ).model_dump(mode="json")
            analysis = None

        if context.operation in (OperationEnum.ANALYZE, OperationEnum.REPORT):
            if analysis is None:
                # При перестроении первая стадия (analysis) уже выставлена в init().
                if not rebuild:
                    await progress.set_stage(Stage.ANALYSIS)
                analysis = await runner.analyze(transcript, params=context.params)
                if analysis is None:
                    await progress.add_warning("analysis_failed")
                else:
                    documents["analysis"] = (
                        await _store(
                            json.dumps(analysis, ensure_ascii=False, indent=2).encode("utf-8"),
                            f"{base}_analysis.json", "application/json", "analysis", context,
                            language_code=analysis["report_lang"], usage=analysis.get("meta"),
                        )
                    ).model_dump(mode="json")
            elif adoc is not None:
                documents["analysis"] = DocumentResponse.model_validate(adoc).model_copy(
                    update={"url": await generate_download_url(adoc.object_key)}
                ).model_dump(mode="json")

        if context.operation == OperationEnum.REPORT:
            await progress.set_stage(Stage.REPORT)
            if analysis is None:
                raise RuntimeError("Report requires analysis, but the LLM analysis failed")
            report_bytes, ext, content_type, warnings = await runner.report(
                transcript, analysis, params=context.params
            )
            for w in warnings:
                await progress.add_warning(w)
            documents["report"] = (
                await _store(report_bytes, f"{base}_report.{ext}", content_type, "report", context,
                             language_code=analysis["report_lang"])
            ).model_dump(mode="json")

        primary = {
            OperationEnum.TRANSCRIBE: "transcript",
            OperationEnum.ANALYZE: "analysis",
            OperationEnum.REPORT: "report",
        }[context.operation]
        main_doc = documents.get(primary) or documents["transcript"]
        await progress.complete(document=main_doc, documents=documents)
        logger.info("task %s complete: %s", context.task_id, list(documents))
    except Exception as e:
        logger.exception("task %s failed", context.task_id)
        await progress.error(details=str(e)[:2000])
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
