from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Settings(BaseSettings):
    endpoint_url: str
    external_url: str

    access_key: str
    secret_key: str
    region: str = "us-east-1"
    use_ssl: bool = True

    storage_bucket: str

    expires_in: int = 3600 * 24

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="S3_",
        case_sensitive=False,
        extra="ignore",
    )
