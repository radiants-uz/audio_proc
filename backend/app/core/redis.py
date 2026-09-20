from arq.connections import RedisSettings
from redis.asyncio import Redis

from app.core.config import settings


def get_arq_redis_settings() -> RedisSettings:
    return RedisSettings(
        host=settings.redis.host,
        port=settings.redis.port,
        password=settings.redis.password.get_secret_value() or None,
    )


def get_redis_client(decode_responses: bool = False) -> Redis:
    return Redis(
        host=settings.redis.host,
        port=settings.redis.port,
        password=settings.redis.password.get_secret_value() or None,
        decode_responses=decode_responses,
    )
