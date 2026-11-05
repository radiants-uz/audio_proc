"""Задача не должна оставаться незавершённой для ARK: таймаут arq, остановка
воркера и отсутствие воркера дают status=error, а не три часа ожидания."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import settings
from app.core.fixtures import OperationEnum
from app.schemas.progress import PROGRESS_PREFIX, WORKER_PREFIX, Status
from app.schemas.task_context import AudioTaskContext
from app.services import audio_processor
from app.services.progress import PENDING_WITHOUT_WORKER_SEC, get_task_progress
from app.workers.arq_worker import new_worker_id
from tests.test_progress_dead_worker import FakeRedis

KEY = f"{PROGRESS_PREFIX}t1"


async def _hang(*_a, **_k):
    await asyncio.sleep(3600)


async def _no_download(*_a, **_k):
    return None


@pytest.fixture
def job(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.audio, "tmp_dir", str(tmp_path))
    monkeypatch.setattr(audio_processor, "download_to_path", _no_download)
    monkeypatch.setattr(audio_processor.PipelineRunner, "transcribe", _hang)
    redis = FakeRedis()
    redis.store[KEY] = {"task_id": "t1", "status": Status.PENDING}
    context = AudioTaskContext(
        task_id="t1", operation=OperationEnum.TRANSCRIBE, source_document_id=1,
        source_object_key="a/input.webm", source_filename="rec.webm",
    )
    return redis, context


def _run(redis, context):
    return audio_processor.process_audio(redis=redis, models=None, context=context, worker_id="host:1:ab")


async def test_job_timeout_reports_error(job, monkeypatch):
    redis, context = job
    monkeypatch.setattr(settings.arq, "job_timeout", 0)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_run(redis, context), 0.05)
    data = redis.store[KEY]
    assert data["status"] == Status.ERROR
    assert "ARQ_JOB_TIMEOUT" in data["details"] and "ingest" in data["details"]


async def test_worker_shutdown_reports_error(job, monkeypatch):
    redis, context = job
    monkeypatch.setattr(settings.arq, "job_timeout", 3600)
    task = asyncio.create_task(_run(redis, context))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    data = redis.store[KEY]
    assert data["status"] == Status.ERROR
    assert "was stopped" in data["details"] and "ARQ_JOB_TIMEOUT" not in data["details"]


async def test_abandoned_task_is_not_started(job):
    redis, context = job
    redis.store[KEY] = {"task_id": "t1", "status": Status.ERROR, "details": "no worker"}
    await _run(redis, context)  # transcribe завис бы: задачу запускать нельзя
    assert redis.store[KEY] == {"task_id": "t1", "status": Status.ERROR, "details": "no worker"}


def test_timeout_is_recognised_just_before_the_limit(monkeypatch):
    monkeypatch.setattr(settings.arq, "job_timeout", 3600)
    # Таймер arq запущен раньше process_audio: на таймауте прошло чуть меньше лимита.
    assert "ARQ_JOB_TIMEOUT" in audio_processor._interrupted_details("asr", 3599.8)
    assert "was stopped" in audio_processor._interrupted_details("asr", 1800)


def _pending(age_sec: float) -> FakeRedis:
    redis = FakeRedis()
    created = datetime.now(UTC) - timedelta(seconds=age_sec)
    redis.store[KEY] = {"task_id": "t1", "status": Status.PENDING, "created_at": created.isoformat()}
    return redis


async def test_pending_without_workers_becomes_error():
    data = await get_task_progress(_pending(PENDING_WITHOUT_WORKER_SEC + 1), "t1")
    assert data["status"] == Status.ERROR and "no audio-proc worker" in data["details"]


async def test_recent_pending_waits_for_a_worker_restart():
    data = await get_task_progress(_pending(PENDING_WITHOUT_WORKER_SEC - 60), "t1")
    assert data["status"] == Status.PENDING


async def test_old_pending_with_live_worker_stays_queued():
    redis = _pending(PENDING_WITHOUT_WORKER_SEC * 10)
    redis.store[f"{WORKER_PREFIX}host:1:ab"] = "1"
    data = await get_task_progress(redis, "t1")
    assert data["status"] == Status.PENDING


def test_worker_id_changes_between_boots():
    assert new_worker_id() != new_worker_id()
