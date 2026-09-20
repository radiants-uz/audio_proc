from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, File, Request, Response, UploadFile, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    async_get_db,
    get_analyze_params,
    get_arq_pool,
    get_redis,
    get_report_params,
    get_transcribe_params,
)
from app.core.fixtures import OperationEnum
from app.core.s3 import generate_download_url
from app.crud.document import get_document
from app.exceptions.base import NotFoundError
from app.schemas.common import TaskID
from app.schemas.document import DocumentResponse
from app.schemas.report_request import RebuildReportRequest
from app.schemas.task_context import TaskParams
from app.services.progress import get_task_progress
from app.services.task_service import start_audio_processing, start_report_rebuild

router = APIRouter()


# Статические пути объявлены раньше /{document_id}, иначе int-параметр их перехватит.
@router.get("/tasks/{task_id}")
async def get_audio_progress(task_id: str, redis: Redis = Depends(get_redis)):
    return await get_task_progress(redis=redis, task_id=task_id)


async def _start(
    request: Request,
    response: Response,
    operation: OperationEnum,
    file: UploadFile,
    params: TaskParams,
    db: AsyncSession,
    redis: Redis,
    arq_pool: ArqRedis,
) -> TaskID:
    task_id = await start_audio_processing(
        file=file, operation=operation, params=params, db=db, redis=redis, arq_pool=arq_pool
    )
    response.headers["Location"] = str(request.url_for("get_audio_progress", task_id=task_id.task_id))
    return task_id


@router.post(f"/{OperationEnum.TRANSCRIBE.value}", response_model=TaskID, status_code=status.HTTP_202_ACCEPTED)
async def start_transcribe(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    params: TaskParams = Depends(get_transcribe_params),
    db: AsyncSession = Depends(async_get_db),
    redis: Redis = Depends(get_redis),
    arq_pool: ArqRedis = Depends(get_arq_pool),
):
    """Транскрипт: спикеры, язык каждого, реплики с таймкодами."""
    return await _start(request, response, OperationEnum.TRANSCRIBE, file, params, db, redis, arq_pool)


@router.post(f"/{OperationEnum.ANALYZE.value}", response_model=TaskID, status_code=status.HTTP_202_ACCEPTED)
async def start_analyze(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    params: TaskParams = Depends(get_analyze_params),
    db: AsyncSession = Depends(async_get_db),
    redis: Redis = Depends(get_redis),
    arq_pool: ArqRedis = Depends(get_arq_pool),
):
    """Транскрипт + анализ LLM (резюме, темы, решения, задачи, статистика)."""
    return await _start(request, response, OperationEnum.ANALYZE, file, params, db, redis, arq_pool)


@router.post(f"/{OperationEnum.REPORT.value}", response_model=TaskID, status_code=status.HTTP_202_ACCEPTED)
async def start_report(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    params: TaskParams = Depends(get_report_params),
    db: AsyncSession = Depends(async_get_db),
    redis: Redis = Depends(get_redis),
    arq_pool: ArqRedis = Depends(get_arq_pool),
):
    """Транскрипт + анализ + отчёт (pdf | docx)."""
    return await _start(request, response, OperationEnum.REPORT, file, params, db, redis, arq_pool)


@router.post("/{document_id}/report", response_model=TaskID, status_code=status.HTTP_202_ACCEPTED)
async def rebuild_report(
    document_id: int,
    body: RebuildReportRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(async_get_db),
    redis: Redis = Depends(get_redis),
    arq_pool: ArqRedis = Depends(get_arq_pool),
):
    """Перестроить отчёт (и анализ, если regenerate_analysis или его ещё нет)
    по готовому транскрипту — document_id транскрипта или исходника."""
    task_id = await start_report_rebuild(
        document_id=document_id, body=body, db=db, redis=redis, arq_pool=arq_pool
    )
    response.headers["Location"] = str(request.url_for("get_audio_progress", task_id=task_id.task_id))
    return task_id


@router.get("/{document_id}", response_model=DocumentResponse)
async def read_document(document_id: int, db: AsyncSession = Depends(async_get_db)):
    doc_orm = await get_document(db=db, doc_id=document_id)
    if not doc_orm:
        raise NotFoundError(f"Document {document_id} not found")
    doc = DocumentResponse.model_validate(doc_orm)
    return doc.model_copy(update={"url": await generate_download_url(doc.object_key)})
