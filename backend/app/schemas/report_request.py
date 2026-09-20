
from pydantic import BaseModel, Field

from app.core.fixtures import ReportFormat


class RebuildReportRequest(BaseModel):
    """POST /v1/audio/{document_id}/report — перестроить отчёт (и анализ)
    по готовому транскрипту без повторного распознавания."""

    model: str | None = None
    report_format: ReportFormat = ReportFormat.PDF
    report_lang: str = "auto"
    title: str | None = None
    speaker_names: dict[str, str] = Field(default_factory=dict)
    regenerate_analysis: bool = False
