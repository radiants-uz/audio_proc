"""API отличает «воркер долго считает» от «воркер убит»: без heartbeat задача
в processing превращается в error, а не висит до TTL (ARK ждал бы 3 часа)."""

import pytest

from app.schemas.progress import PROGRESS_PREFIX, WORKER_PREFIX, Status
from app.services.progress import Progress, get_task_progress


class _FakeJson:
    def __init__(self, store):
        self.store = store

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, path, value):
        p = str(path).strip(".") if not isinstance(path, str) else path.strip(".")
        if p in ("", "$"):
            self.store[key] = value
        else:
            self.store.setdefault(key, {})[p] = value

    async def numincrby(self, key, path, n):
        p = str(path).strip(".")
        self.store[key][p] = self.store[key].get(p, 0) + n


class FakeRedis:
    def __init__(self):
        self.store: dict = {}
        self.expired: list = []

    def json(self):
        return _FakeJson(self.store)

    async def exists(self, key):
        return 1 if key in self.store else 0

    async def expire(self, key, ttl):
        self.expired.append((key, ttl))


@pytest.mark.asyncio
async def test_processing_task_without_heartbeat_becomes_error():
    redis = FakeRedis()
    redis.store[f"{PROGRESS_PREFIX}t1"] = {"task_id": "t1", "status": Status.PENDING}
    await Progress(redis, "t1").init(total_stages=7, stage="ingest", worker_id="host:1")
    assert redis.store[f"{PROGRESS_PREFIX}t1"]["worker"] == "host:1"

    redis.store[f"{WORKER_PREFIX}host:1"] = "1"  # heartbeat жив
    data = await get_task_progress(redis, "t1")
    assert data["status"] == Status.PROCESSING

    del redis.store[f"{WORKER_PREFIX}host:1"]  # воркер убит, heartbeat истёк
    data = await get_task_progress(redis, "t1")
    assert data["status"] == Status.ERROR and "died during stage 'ingest'" in data["details"]


@pytest.mark.asyncio
async def test_legacy_record_without_worker_is_left_alone():
    redis = FakeRedis()
    redis.store[f"{PROGRESS_PREFIX}t2"] = {"task_id": "t2", "status": Status.PROCESSING, "stage": "asr"}
    data = await get_task_progress(redis, "t2")
    assert data["status"] == Status.PROCESSING
