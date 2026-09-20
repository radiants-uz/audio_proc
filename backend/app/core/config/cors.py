
from pydantic_settings import BaseSettings, SettingsConfigDict


class CorsSettings(BaseSettings):
    allow_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    allow_credentials: bool = True
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CORS_",
        case_sensitive=False,
        extra="ignore",
    )
