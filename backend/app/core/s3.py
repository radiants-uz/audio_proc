import re
import uuid
from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import urlparse, urlunparse

import aioboto3
from fastapi import UploadFile

from app.core.config import settings
from app.core.stored_file import StoredFile

# Разделители путей и управляющие символы: всё, чем можно уехать из префикса
# ключа или сломать его. Буквы любых алфавитов оставляем как есть.
_UNSAFE_KEY_CHARS = re.compile(r"[\x00-\x1f\x7f/\\]+")
_MAX_KEY_NAME_LEN = 200


def safe_object_name(filename: str | None) -> str:
    """Имя файла, пригодное для ключа S3 (берётся только базовая часть)."""
    name = str(filename or "")
    name = PureWindowsPath(PurePosixPath(name).name).name
    name = _UNSAFE_KEY_CHARS.sub("_", name)
    name = name.lstrip(".").strip()
    return name[:_MAX_KEY_NAME_LEN] or "file"


def get_s3_client():
    """Возвращает контекстный менеджер для S3-клиента."""
    session = aioboto3.Session(
        aws_access_key_id=settings.s3.access_key,
        aws_secret_access_key=settings.s3.secret_key,
        region_name=settings.s3.region,
    )
    return session.client(
        "s3",
        endpoint_url=settings.s3.endpoint_url,
        use_ssl=settings.s3.use_ssl,
        verify=settings.s3.use_ssl,
    )


def make_object_key(filename: str | None, prefix: str = "audio", folder: str | None = None) -> str:
    """`audio/{YYYY-MM-DD}/{uuid}/{name}`. Один `folder` на задачу — исходник и
    все результаты лежат рядом."""
    folder = folder or str(uuid.uuid4())
    return f"{prefix}/{datetime.now().strftime('%Y-%m-%d')}/{folder}/{safe_object_name(filename)}"


def folder_of(object_key: str) -> str | None:
    parts = object_key.split("/")
    return parts[2] if len(parts) >= 4 else None


async def upload_to_s3(
    fileobj: StoredFile | UploadFile,
    prefix: str = "audio",
    bucket: str | None = None,
    folder: str | None = None,
) -> str:
    object_key = make_object_key(fileobj.filename, prefix=prefix, folder=folder)
    bucket = bucket or settings.s3.storage_bucket

    async with get_s3_client() as s3:
        await s3.upload_fileobj(
            Fileobj=fileobj.file,
            Bucket=bucket,
            Key=object_key,
            ExtraArgs={"ContentType": fileobj.content_type},
        )

    return object_key


async def download_to_path(object_key: str, path: str, bucket: str | None = None) -> None:
    """Скачать объект на локальный диск потоково (исходник может быть до 500 МБ)."""
    bucket = bucket or settings.s3.storage_bucket
    async with get_s3_client() as s3:
        await s3.download_file(bucket, object_key, path)


async def get_file_from_s3(object_key: str, bucket: str | None = None) -> StoredFile:
    bucket = bucket or settings.s3.storage_bucket
    async with get_s3_client() as s3:
        response = await s3.get_object(Bucket=bucket, Key=object_key)
        content = await response["Body"].read()
        content_type = response.get("ContentType")

    return StoredFile(
        data=content,
        filename=object_key.split("/")[-1],
        content_type=content_type,
    )


async def delete_objects(object_keys: list[str], bucket: str | None = None) -> None:
    if not object_keys:
        return
    bucket = bucket or settings.s3.storage_bucket
    async with get_s3_client() as s3:
        await s3.delete_objects(
            Bucket=bucket, Delete={"Objects": [{"Key": k} for k in object_keys], "Quiet": True}
        )


async def check_bucket(bucket: str | None = None) -> None:
    bucket = bucket or settings.s3.storage_bucket
    async with get_s3_client() as s3:
        await s3.head_bucket(Bucket=bucket)


async def generate_download_url(
    object_key: str,
    bucket: str | None = None,
    expires_in: int | None = None,
) -> str:
    bucket = bucket or settings.s3.storage_bucket
    expires_in = expires_in or settings.s3.expires_in
    async with get_s3_client() as s3:
        url = await s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": object_key},
            ExpiresIn=expires_in,
        )

    # Presigned-ссылка подписана для внутреннего хоста (minio:9000); наружу
    # отдаём хост из S3_EXTERNAL_URL — MinIO проверяет только путь и query.
    parsed = urlparse(url)
    external_parsed = urlparse(settings.s3.external_url)
    new_url = parsed._replace(
        scheme=external_parsed.scheme or parsed.scheme, netloc=external_parsed.netloc
    )
    return urlunparse(new_url)
