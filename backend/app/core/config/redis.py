from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    password: SecretStr = SecretStr("")
    # Имя переменной с опечаткой унаследовано от doc-proc (REDIS_PROGRES_TTL),
    # чтобы .env обоих сервисов выглядели одинаково.
    progres_ttl: int = 3600 * 24

    @property
    def progress_ttl(self) -> int:
        return self.progres_ttl

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="REDIS_",
        case_sensitive=False,
        extra="ignore",
    )
