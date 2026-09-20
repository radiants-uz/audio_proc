from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .admin import Admin
from .arq import ArqSettings
from .audio import (
    AnalysisSettings,
    AsrSettings,
    AudioSettings,
    DiarizationSettings,
    LangIdSettings,
    ReportSettings,
    SegmentsSettings,
)
from .cors import CorsSettings
from .database import DBSettings
from .redis import RedisSettings
from .s3 import S3Settings


class Settings(BaseSettings):
    s3: S3Settings = S3Settings()
    db: DBSettings = DBSettings()
    redis: RedisSettings = RedisSettings()
    arq: ArqSettings = ArqSettings()
    cors: CorsSettings = CorsSettings()
    admin: Admin = Admin()

    audio: AudioSettings = AudioSettings()
    diarization: DiarizationSettings = DiarizationSettings()
    segments: SegmentsSettings = SegmentsSettings()
    langid: LangIdSettings = LangIdSettings()
    asr: AsrSettings = AsrSettings()
    analysis: AnalysisSettings = AnalysisSettings()
    report: ReportSettings = ReportSettings()

    max_file_size_mb: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BACKEND_",
        case_sensitive=False,
        extra="ignore",
    )

    @computed_field
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()
