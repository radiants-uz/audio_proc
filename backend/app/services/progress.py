"""Прогресс задачи в Redis-JSON. Форма полей — как в doc-proc, чтобы
ARK читал её тем же кодом (§6.1 плана)."""

import re
import uuid
from datetime import UTC, datetime

from redis.asyncio import Redis
from redis.commands.json.path import Path

from app.core.config import settings
from app.exceptions.base import TaskIDNotFoundError
from app.schemas.progress import PROGRESS_PREFIX, Stage, Status


class Progress:
    def __init__(self, redis: Redis, task_id: str):
        self._redis = redis
        self.task_id = task_id
        self._key = f"{PROGRESS_PREFIX}{task_id}"
        self._ttl = settings.redis.progress_ttl

    async def exists(self) -> bool:
        return bool(await self._redis.json().get(self._key))

    async def init(self, total_stages: int, stage: str = Stage.INGEST):
        if not await self.exists():
            raise TaskIDNotFoundError(
                f"Task ID '{self.task_id}' not found in Redis. Was it created by the API?"
            )
        now = datetime.now(UTC).isoformat()
        await self._set(".status", Status.PROCESSING)
        await self._set(".stage", stage)
        await self._set(".total_stages", total_stages)
        # Первая стадия уже идёт: current_stage/total_stages = 1/N.
        await self._set(".current_stage", 1)
        await self._set(".total", 1)
        await self._set(".current", 0)
        await self._set(".started_at", now)

    async def set_stage(self, stage: str, total: int = 1):
        """Следующая стадия: current_stage += 1, счётчик внутри стадии сбрасывается."""
        await self._set(".stage", stage)
        await self._redis.json().numincrby(self._key, Path(".current_stage"), 1)
        await self._set(".total", max(int(total), 1))
        await self._set(".current", 0)

    async def set_total(self, total: int):
        await self._set(".total", max(int(total), 1))

    async def increment_current(self, step: int = 1):
        await self._redis.json().numincrby(self._key, Path(".current"), step)

    async def set_current(self, current: int):
        await self._set(".current", int(current))

    async def add_warning(self, warning: str):
        data = await self.get() or {}
        warnings = list(data.get("warnings") or [])
        if warning not in warnings:
            warnings.append(warning)
        await self._set(".warnings", warnings)

    async def complete(self, document: dict, documents: dict):
        await self._set(".status", Status.COMPLETE)
        await self._set(".completed_at", datetime.now(UTC).isoformat())
        await self._set(".processed_document", document)
        await self._set(".documents", documents)
        await self._redis.expire(self._key, self._ttl)

    async def error(self, details: str):
        # Текст ошибки уходит в ARK как есть: без управляющих символов и без
        # многокилобайтных SQL из трейсбеков.
        details = re.sub(r"[\x00-\x1f]+", " ", str(details)).strip()[:2000]
        await self._set(".status", Status.ERROR)
        await self._set(".completed_at", datetime.now(UTC).isoformat())
        await self._set(".details", details)
        await self._redis.expire(self._key, self._ttl)

    async def get(self) -> dict | None:
        return await self._redis.json().get(self._key)

    async def _set(self, path: str, value):
        await self._redis.json().set(self._key, Path(path), value)


async def init_task_progress(redis: Redis, meta: dict | None = None) -> str:
    task_id = str(uuid.uuid4())
    key = f"{PROGRESS_PREFIX}{task_id}"
    payload = {
        "task_id": task_id,
        "status": Status.PENDING,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if meta:
        payload["meta"] = meta
    await redis.json().set(key, Path.root_path(), payload)
    return task_id


async def get_task_progress(redis: Redis, task_id: str) -> dict:
    data = await redis.json().get(f"{PROGRESS_PREFIX}{task_id}")
    if not data:
        raise TaskIDNotFoundError(detail=f"Task ID: '{task_id}' not found.")
    return data
