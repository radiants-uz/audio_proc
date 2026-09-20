"""Приём задачи в api: валидация → исходник в S3 → запись Document →
прогресс в Redis → задача в очередь arq."""

import logging

from arq.connections import ArqRedis
from fastapi import UploadFile
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.fixtures import AUDIO_EXTENSIONS, OperationEnum
from app.core.s3 import upload_to_s3
from app.crud.document import create_document, get_children, get_document
from app.crud.service import get_service
from app.exceptions.base import ClientAppError, FileTooLargeError, NotFoundError, ServerAppError
from app.models.document import Document
from app.schemas.common import TaskID
from app.schemas.report_request import RebuildReportRequest
from app.schemas.task_context import AudioTaskContext, TaskParams
from app.services.progress import init_task_progress

logger = logging.getLogger(__name__)

TASK_FUNCTION = "process_audio_task"


def _extension(filename: str | None) -> str:
    name = filename or ""
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


async def _validate_llm_service(db: AsyncSession, operation: OperationEnum, params: TaskParams) -> None:
    if operation == OperationEnum.TRANSCRIBE:
        return
    if not params.model:
        raise ClientAppError("model is required for analyze/report")
    service = await get_service(db, params.model)
    if not service or not service.is_active:
        raise ClientAppError(f"LLM service '{params.model}' not found. See GET /v1/services")
    if service.operations and operation.value not in service.operations:
        raise ClientAppError(f"LLM service '{params.model}' does not support '{operation.value}'")
    if params.report_lang != "auto" and service.languages and params.report_lang not in service.languages:
        raise ClientAppError(f"LLM service '{params.model}' does not support report_lang='{params.report_lang}'")


async def start_audio_processing(
    file: UploadFile,
    operation: OperationEnum,
    params: TaskParams,
    db: AsyncSession,
    redis: Redis,
    arq_pool: ArqRedis,
) -> TaskID:
    if file.size and file.size > settings.max_file_size_bytes:
        raise FileTooLargeError(f"File too large: limit is {settings.max_file_size_mb} MB")

    ext = _extension(file.filename)
    if not ext:
        raise ClientAppError("File must have an extension")
    if ext not in AUDIO_EXTENSIONS:
        raise ClientAppError(
            f"Unsupported file type '.{ext}'. Supported: {', '.join(AUDIO_EXTENSIONS)}"
        )

    await _validate_llm_service(db, operation, params)

    try:
        object_key = await upload_to_s3(file)
    except Exception as e:  # noqa: BLE001
        raise ServerAppError(f"Error with uploading to s3: {e}") from e

    try:
        source: Document = await create_document(
            db,
            name=str(file.filename),
            object_key=object_key,
            kind="source",
            operation_name=operation.value,
            service_name=params.model,
        )
    except Exception as e:  # noqa: BLE001
        raise ServerAppError(f"Error with creating document: {e}") from e

    task_id = await init_task_progress(redis, meta={"operation": operation.value})

    context = AudioTaskContext(
        task_id=task_id,
        operation=operation,
        source_document_id=source.id,
        source_object_key=source.object_key,
        source_filename=source.name,
        params=params,
    )
    await arq_pool.enqueue_job(TASK_FUNCTION, context.model_dump(mode="json"))
    logger.info("enqueued %s task %s (document %s)", operation.value, task_id, source.id)
    return TaskID(task_id=task_id)


async def start_report_rebuild(
    document_id: int,
    body: RebuildReportRequest,
    db: AsyncSession,
    redis: Redis,
    arq_pool: ArqRedis,
) -> TaskID:
    """document_id — транскрипт или исходник; ищем последний transcript.json
    и последний analysis.json рядом с ним."""
    doc = await get_document(db, document_id)
    if not doc:
        raise NotFoundError(f"Document {document_id} not found")

    if doc.kind == "transcript":
        transcript, source_id = doc, doc.parent_id
    else:
        source_id = doc.id if doc.kind == "source" else doc.parent_id
        transcripts = await get_children(db, source_id, kind="transcript") if source_id else []
        if not transcripts:
            raise ClientAppError(f"Document {document_id} has no transcript yet")
        transcript = transcripts[0]

    analyses = await get_children(db, source_id, kind="analysis") if source_id else []
    analysis = analyses[0] if analyses and not body.regenerate_analysis else None

    params = TaskParams(
        model=body.model,
        report_format=body.report_format.value,
        report_lang=body.report_lang,
        title=body.title,
        speaker_names=body.speaker_names,
        regenerate_analysis=body.regenerate_analysis,
    )
    # Анализ придётся считать → нужен LLM-сервис.
    if analysis is None:
        await _validate_llm_service(db, OperationEnum.REPORT, params)

    source = await get_document(db, source_id) if source_id else None
    task_id = await init_task_progress(redis, meta={"operation": OperationEnum.REPORT.value, "rebuild": True})
    context = AudioTaskContext(
        task_id=task_id,
        operation=OperationEnum.REPORT,
        source_document_id=source_id,
        source_object_key=source.object_key if source else None,
        source_filename=source.name if source else transcript.name,
        transcript_document_id=transcript.id,
        analysis_document_id=analysis.id if analysis else None,
        params=params,
    )
    await arq_pool.enqueue_job(TASK_FUNCTION, context.model_dump(mode="json"))
    return TaskID(task_id=task_id)
