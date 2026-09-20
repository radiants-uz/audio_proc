"""Воркер arq. Модели (диаризация, LID) грузятся один раз в on_startup —
иначе каждая задача ждала бы загрузку 20–40 с. Одна задача на процесс."""

import asyncio
import logging
import os
import socket

from redis.asyncio import Redis

from app.core.config import settings
from app.core.redis import get_arq_redis_settings, get_redis_client
from app.schemas.progress import WORKER_PREFIX
from app.schemas.task_context import AudioTaskContext
from app.services.audio_processing.models import ModelRegistry
from app.services.audio_processor import process_audio

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

HEARTBEAT_TTL = 30


async def process_audio_task(ctx, context: dict):
    redis: Redis = ctx["redis"]
    models: ModelRegistry = ctx["models"]
    await process_audio(redis=redis, models=models, context=AudioTaskContext.model_validate(context))


async def _heartbeat_loop(redis: Redis, worker_id: str):
    while True:
        try:
            await redis.set(f"{WORKER_PREFIX}{worker_id}", "1", ex=HEARTBEAT_TTL)
        except Exception as e:  # noqa: BLE001
            logger.warning("heartbeat failed: %s", e)
        await asyncio.sleep(HEARTBEAT_TTL // 3)


async def startup(ctx):
    ctx["redis"] = get_redis_client(decode_responses=True)
    ctx["worker_id"] = f"{socket.gethostname()}:{os.getpid()}"
    ctx["heartbeat"] = asyncio.create_task(_heartbeat_loop(ctx["redis"], ctx["worker_id"]))

    models = ModelRegistry()
    # Загрузка тяжёлая и синхронная — в thread, чтобы не блокировать heartbeat.
    await asyncio.to_thread(models.load_all)
    ctx["models"] = models
    logger.info(
        "worker %s ready: diarization=%s device=%s asr=%s",
        ctx["worker_id"], settings.diarization.backend, settings.audio.device, settings.asr.backend,
    )


async def shutdown(ctx):
    hb = ctx.get("heartbeat")
    if hb:
        hb.cancel()
    redis: Redis = ctx["redis"]
    try:
        await redis.delete(f"{WORKER_PREFIX}{ctx['worker_id']}")
    finally:
        await redis.aclose()


class WorkerSettings:
    functions = [process_audio_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = get_arq_redis_settings()
    max_jobs = settings.arq.max_workers
    job_timeout = settings.arq.job_timeout
    max_tries = settings.arq.max_tries
    # Прогресс в Redis обновляем сами; результат задачи arq не нужен.
    keep_result = 0
    cron_jobs: list = []
