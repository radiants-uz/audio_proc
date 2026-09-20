from urllib.parse import quote_plus

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseSettings):
    HOST: str = "localhost"
    PORT: int = 5432
    BACKEND_USER: str = "backend_user"
    BACKEND_PASSWORD: SecretStr = SecretStr("")
    BACKEND_DB: str = "backend_db"

    BACKEND_MIGRATOR_USER: str = ""
    BACKEND_MIGRATOR_PASSWORD: SecretStr = SecretStr("")

    ECHO: bool = False

    @property
    def db_url(self) -> str:
        password = quote_plus(self.BACKEND_PASSWORD.get_secret_value())
        return f"postgresql+asyncpg://{self.BACKEND_USER}:{password}@{self.HOST}:{self.PORT}/{self.BACKEND_DB}"

    @property
    def migrate_url(self) -> str:
        password = quote_plus(self.BACKEND_MIGRATOR_PASSWORD.get_secret_value())
        return f"postgresql+psycopg2://{self.BACKEND_MIGRATOR_USER}:{password}@{self.HOST}:{self.PORT}/{self.BACKEND_DB}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="POSTGRES_",
        case_sensitive=False,
        extra="ignore",
    )
