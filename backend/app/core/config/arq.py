from pydantic_settings import BaseSettings, SettingsConfigDict


class ArqSettings(BaseSettings):
    # Одна задача на процесс: модели диаризации/LID живут в памяти воркера.
    max_workers: int = 1
    job_timeout: int = 3600
    max_tries: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ARQ_",
        case_sensitive=False,
        extra="ignore",
    )
