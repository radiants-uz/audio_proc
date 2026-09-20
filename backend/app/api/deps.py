import json
import secrets

from fastapi import Form, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.core.database import async_session
from app.core.fixtures import OperationEnum, ReportFormat
from app.exceptions.base import ClientAppError
from app.schemas.common import ServiceName
from app.schemas.task_context import TaskParams


async def get_redis(request: Request):
    return request.app.state.redis


async def get_arq_pool(request: Request):
    return request.app.state.arq_pool


async def async_get_db():
    async with async_session() as session:
        yield session


# auto_error=False — ответ на отсутствующий ключ формируем сами (401, а не 403),
# заодно ключ появляется в OpenAPI и в кнопке Authorize на /docs.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_admin_api_key(api_key: str | None = Security(api_key_header)) -> bool:
    expected = settings.admin.API_KEY.get_secret_value()

    # Fail closed: без настроенного ключа API не открывается, а отвечает ошибкой.
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BACKEND_ADMIN_API_KEY is not configured",
        )

    # compare_digest вместо ==, чтобы ключ нельзя было подобрать по времени ответа.
    if not api_key or not secrets.compare_digest(api_key.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
    return True


def _parse_speaker_names(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ClientAppError(f"speaker_names must be a JSON object: {e}") from e
    if not isinstance(data, dict):
        raise ClientAppError("speaker_names must be a JSON object like {\"SPEAKER_00\": \"Name\"}")
    return {str(k): str(v) for k, v in data.items()}


def _check_report_lang(report_lang: str) -> str:
    report_lang = (report_lang or "auto").lower()
    if report_lang not in ("auto", "uz", "ru", "en"):
        raise ClientAppError("report_lang must be one of: auto, uz, ru, en")
    return report_lang


def get_transcribe_params(
    expected_speakers: int | None = Form(default=None, ge=1, le=30),
    lid_recheck: bool | None = Form(default=None),
    speaker_names: str | None = Form(default=None),
    title: str | None = Form(default=None, max_length=255),
) -> TaskParams:
    return TaskParams(
        expected_speakers=expected_speakers,
        lid_recheck=lid_recheck,
        speaker_names=_parse_speaker_names(speaker_names),
        title=title,
    )


def get_analyze_params(
    model: ServiceName = Form(...),
    report_lang: str = Form("auto"),
    expected_speakers: int | None = Form(default=None, ge=1, le=30),
    lid_recheck: bool | None = Form(default=None),
    speaker_names: str | None = Form(default=None),
    title: str | None = Form(default=None, max_length=255),
) -> TaskParams:
    return TaskParams(
        model=model,
        report_lang=_check_report_lang(report_lang),
        expected_speakers=expected_speakers,
        lid_recheck=lid_recheck,
        speaker_names=_parse_speaker_names(speaker_names),
        title=title,
    )


def get_report_params(
    model: ServiceName = Form(...),
    report_format: ReportFormat = Form(ReportFormat.PDF),
    report_lang: str = Form("auto"),
    expected_speakers: int | None = Form(default=None, ge=1, le=30),
    lid_recheck: bool | None = Form(default=None),
    speaker_names: str | None = Form(default=None),
    title: str | None = Form(default=None, max_length=255),
) -> TaskParams:
    return TaskParams(
        model=model,
        report_format=report_format.value,
        report_lang=_check_report_lang(report_lang),
        expected_speakers=expected_speakers,
        lid_recheck=lid_recheck,
        speaker_names=_parse_speaker_names(speaker_names),
        title=title,
    )


PARAMS_BY_OPERATION = {
    OperationEnum.TRANSCRIBE: get_transcribe_params,
    OperationEnum.ANALYZE: get_analyze_params,
    OperationEnum.REPORT: get_report_params,
}
