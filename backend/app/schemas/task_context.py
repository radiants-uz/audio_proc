"""Контекст задачи — то, что api кладёт в очередь arq, а воркер исполняет."""


from pydantic import BaseModel, Field

from app.core.fixtures import OperationEnum


class TaskParams(BaseModel):
    expected_speakers: int | None = Field(None, ge=1, le=30)
    lid_recheck: bool | None = None
    speaker_names: dict[str, str] = Field(default_factory=dict)
    title: str | None = None
    # analyze / report
    model: str | None = None
    report_lang: str = "auto"
    report_format: str = "pdf"
    regenerate_analysis: bool = False


class AudioTaskContext(BaseModel):
    task_id: str
    operation: OperationEnum
    # Исходное аудио (kind=source). Для перестроения отчёта — None.
    source_document_id: int | None = None
    source_object_key: str | None = None
    source_filename: str = "audio"
    # Перестроение по готовому транскрипту (POST /v1/audio/{id}/report).
    transcript_document_id: int | None = None
    analysis_document_id: int | None = None
    params: TaskParams = Field(default_factory=TaskParams)
