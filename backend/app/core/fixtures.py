"""Перечисления, общие для API, воркера и ARK-клиента."""

from enum import StrEnum


class OperationEnum(StrEnum):
    # Значения — сегменты URL: POST /v1/audio/{operation}
    TRANSCRIBE = "transcribe"
    ANALYZE = "analyze"
    REPORT = "report"


class ProviderEnum(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"


class ReportFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"


class LanguageCode(StrEnum):
    AUTO = "auto"
    UZ = "uz"
    RU = "ru"
    EN = "en"


# Что принимаем на вход. Расширение проверяется по имени файла, контейнер
# разбирает ffmpeg (видео тоже — берётся аудиодорожка).
AUDIO_EXTENSIONS = (
    "wav", "mp3", "m4a", "aac", "ogg", "oga", "opus", "flac", "wma", "amr", "webm",
    "mp4", "mkv", "mov", "3gp", "mpeg", "mpga",
)

# Сколько стадий показывает прогресс для каждой операции (§6.1).
STAGES_BY_OPERATION = {
    OperationEnum.TRANSCRIBE: ("ingest", "diarization", "langid", "asr", "assemble"),
    OperationEnum.ANALYZE: ("ingest", "diarization", "langid", "asr", "assemble", "analysis"),
    OperationEnum.REPORT: ("ingest", "diarization", "langid", "asr", "assemble", "analysis", "report"),
}

# Перестроение отчёта по готовому транскрипту (POST /v1/audio/{id}/report).
REBUILD_STAGES = ("analysis", "report")
