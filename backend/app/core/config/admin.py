from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Admin(BaseSettings):
    USER: str = "admin"
    PASSWORD: SecretStr = SecretStr("")
    API_KEY: SecretStr = SecretStr("")

    # Ключ подписи сессионной куки админки. Без дефолта намеренно:
    # пустой/предсказуемый ключ позволяет подделать сессию админа.
    SECRET_KEY: SecretStr

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BACKEND_ADMIN_",
        case_sensitive=False,
        extra="ignore",
    )
