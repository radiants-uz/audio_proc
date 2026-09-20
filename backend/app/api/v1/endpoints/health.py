"""GET /v1/health/deep — под ключом: состояние зависимостей и воркеров."""

import asyncio
import logging

import httpx
from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import async_get_db, get_redis
from app.core.config import settings
from app.core.s3 import check_bucket
from app.schemas.progress import WORKER_PREFIX

router = APIRouter()
logger = logging.getLogger(__name__)


async def _check(coro) -> dict:
    try:
        await asyncio.wait_for(coro, timeout=10)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:300]}


async def _check_asr():
    if settings.asr.backend == "mock":
        return
    async with httpx.AsyncClient(timeout=10) as client:
        # Любой ответ приложения (даже 405/422 на GET) — сервис жив; 5xx — нет.
        resp = await client.get(settings.asr.api_url)
        if resp.status_code >= 500:
            raise RuntimeError(f"ASR API returned HTTP {resp.status_code}")


@router.get("/deep")
async def deep_health(redis: Redis = Depends(get_redis), db: AsyncSession = Depends(async_get_db)):
    async def _db():
        await db.execute(text("SELECT 1"))

    async def _redis():
        await redis.ping()

    redis_ok, db_ok, s3_ok, asr_ok = await asyncio.gather(
        _check(_redis()), _check(_db()), _check(check_bucket()), _check(_check_asr())
    )

    workers = []
    queue_len = 0
    try:
        async for key in redis.scan_iter(f"{WORKER_PREFIX}*"):
            workers.append(key.split(":", 1)[1])
        queue_len = await redis.zcard("arq:queue")
    except Exception as e:  # noqa: BLE001
        logger.warning("worker/queue check failed: %s", e)

    return {
        "status": "ok" if all(x["ok"] for x in (redis_ok, db_ok, s3_ok)) else "degraded",
        "redis": redis_ok,
        "db": db_ok,
        "s3": s3_ok,
        "asr_api": {**asr_ok, "backend": settings.asr.backend},
        "diarization_backend": settings.diarization.backend,
        "device": settings.audio.device,
        "workers_alive": len(workers),
        "workers": workers,
        "queue_length": queue_len,
    }
