"""Правила общего GLM-сервера для клиента анализа (без сети): не больше 8
одновременных запросов с ключа, паузы 5/10/20 с и не больше 3 повторов,
таймаут ≤ 300 с, никаких повторов «в лоб»."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.core.config import settings
from app.models.service import Service
from app.services.audio_processing.analysis import llm_openai
from app.services.audio_processing.analysis.llm_openai import (
    LLM_MAX_TIMEOUT,
    LlmError,
    OpenAICompatibleClient,
    key_semaphore,
)

REQUEST = httpx.Request("POST", "http://gw/v1/chat/completions")


def _status_error(status):
    return APIStatusError("boom", response=httpx.Response(status, request=REQUEST), body=None)


def _answer(content='{"ok": true}'):
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2)
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _service(**kw):
    kw.setdefault("limit_requests", 2)
    return Service(name="glm", base_url="http://gw/v1", api_key="test-key", model_name="glm-5.3", **kw)


def _client(*outcomes, **service_kw):
    """Клиент, у которого chat.completions.create по очереди бросает/возвращает заданное."""
    client = OpenAICompatibleClient(_service(**service_kw))
    client.calls = []
    outcomes = list(outcomes)

    async def create(**kwargs):
        client.calls.append(kwargs)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client


@pytest.fixture
def pauses(monkeypatch):
    """Паузы не ждём, а записываем."""
    recorded = []

    async def fake_sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(llm_openai, "sleep", fake_sleep)
    return recorded


def _complete(client):
    return client.complete_json("sys", "user", {"type": "object"}, 100)


# --- паузы и число повторов ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_retryable_status_waits_5_10_20_and_gives_up_after_3_retries(pauses, status):
    client = _client(*[_status_error(status) for _ in range(10)])
    with pytest.raises(LlmError, match="after 4 attempts"):
        await _complete(client)
    assert len(client.calls) == 4  # первая отправка + 3 повтора, не больше
    # Каждый раз дольше предыдущей: 5, 10, 20 с (+ до 1 с разброса).
    assert [int(p) for p in pauses] == [5, 10, 20]
    assert [5 <= pauses[0] < 6, 10 <= pauses[1] < 11, 20 <= pauses[2] < 21] == [True] * 3


@pytest.mark.asyncio
async def test_connection_error_and_timeout_are_retried_with_pauses(pauses):
    client = _client(APIConnectionError(request=REQUEST), APITimeoutError(request=REQUEST), _answer())
    resp = await _complete(client)
    assert resp.data == {"ok": True} and len(client.calls) == 3 and len(pauses) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404])
async def test_client_errors_are_not_repeated(pauses, status):
    client = _client(_status_error(status), _answer())
    with pytest.raises(LlmError, match=f"HTTP {status}"):
        await _complete(client)
    assert len(client.calls) == 1 and pauses == []


@pytest.mark.asyncio
async def test_empty_answer_is_repeated_only_after_pauses(pauses):
    client = _client(_answer(""), _answer("   "), _answer())
    resp = await _complete(client)
    assert resp.data == {"ok": True} and len(client.calls) == 3
    assert len(pauses) == 2 and pauses[0] < pauses[1]  # не сразу и всё дольше


@pytest.mark.asyncio
async def test_rejected_response_format_falls_back_without_a_pause(pauses):
    """400 на response_format — другой запрос (без него), а не повтор: пауза не нужна."""
    client = _client(_status_error(400), _answer())
    resp = await _complete(client)
    assert resp.data == {"ok": True}
    assert "response_format" in client.calls[0] and "response_format" not in client.calls[1]
    assert pauses == []


@pytest.mark.asyncio
async def test_retry_limits_come_from_settings(pauses, monkeypatch):
    monkeypatch.setattr(settings.analysis, "retries", 1)
    monkeypatch.setattr(settings.analysis, "retry_base_delay", 7.0)
    client = _client(*[_status_error(503) for _ in range(5)])
    with pytest.raises(LlmError):
        await _complete(client)
    assert len(client.calls) == 2 and 7 <= pauses[0] < 8 and len(pauses) == 1


# --- клиент и таймаут ---------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("configured, expected", [(600.0, 300.0), (120.0, 120.0), (300.0, 300.0)])
async def test_client_timeout_never_exceeds_300_seconds(monkeypatch, configured, expected):
    monkeypatch.setattr(settings.analysis, "timeout_sec", configured)
    client = OpenAICompatibleClient(_service())
    assert client._client.timeout == expected <= LLM_MAX_TIMEOUT


@pytest.mark.asyncio
async def test_sdk_does_not_retry_on_its_own():
    # Иначе SDK повторял бы без пауз, перемножаясь с нашими повторами.
    assert OpenAICompatibleClient(_service())._client.max_retries == 0


# --- не больше 8 запросов с ключа -----------------------------------------------


async def _peak(limit, n_tasks=40, key="k", clients=3):
    sems = [key_semaphore("http://gw/v1", key, limit) for _ in range(clients)]  # «клиенты разных задач»
    assert all(s is sems[0] for s in sems)
    state = {"now": 0, "peak": 0}

    async def one(i):
        async with sems[i % clients]:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            await asyncio.sleep(0.001)
            state["now"] -= 1

    await asyncio.gather(*(one(i) for i in range(n_tasks)))
    return state["peak"]


@pytest.mark.asyncio
async def test_one_key_never_exceeds_8_even_if_service_asks_for_100():
    assert await _peak(limit=100) == 8


@pytest.mark.asyncio
async def test_service_limit_can_lower_the_cap():
    assert await _peak(limit=3, key="other") == 3


@pytest.mark.asyncio
async def test_cap_comes_from_settings(monkeypatch):
    monkeypatch.setattr(settings.analysis, "max_concurrent_requests", 2)
    assert await _peak(limit=100, key="third") == 2


@pytest.mark.asyncio
async def test_clients_of_different_tasks_share_the_key_semaphore():
    a = OpenAICompatibleClient(_service(limit_requests=100))
    b = OpenAICompatibleClient(_service(limit_requests=100))
    assert a._sem is b._sem


@pytest.mark.asyncio
async def test_different_keys_do_not_share_a_semaphore():
    assert key_semaphore("http://gw/v1", "key-a", 8) is not key_semaphore("http://gw/v1", "key-b", 8)


def test_semaphore_is_recreated_for_a_new_event_loop():
    async def get():
        return key_semaphore("http://gw/v1", "loop-key", 8)

    assert asyncio.run(get()) is not asyncio.run(get())
