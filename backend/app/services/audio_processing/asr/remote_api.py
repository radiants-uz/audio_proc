"""§5.5: клиент существующего ASR-API заказчика.

POST {ASR_API_URL}, multipart: `file` (wav 16k mono ≤ 30 с) + `language_code`
(uz|ru|en). Ответ JSON, читаем поле `transcription`. Без авторизации
(опционально ASR_AUTH_HEADER/ASR_AUTH_TOKEN)."""

import asyncio
import logging
import random

import httpx

from app.core.config import settings
from app.services.audio_processing.asr.base import AsrResult

logger = logging.getLogger(__name__)


class AsrApiError(Exception):
    pass


class RemoteAsrClient:
    def __init__(self):
        cfg = settings.asr
        self.name = f"remote_api({cfg.api_url})"
        headers = {}
        if cfg.auth_header and cfg.auth_token.get_secret_value():
            headers[cfg.auth_header] = cfg.auth_token.get_secret_value()
        self._client = httpx.AsyncClient(timeout=cfg.timeout_sec, headers=headers)
        self._sem = asyncio.Semaphore(max(1, cfg.concurrency))

    async def transcribe(self, lang: str, wav_bytes: bytes) -> AsrResult:
        cfg = settings.asr
        last_error: Exception | None = None
        for attempt in range(cfg.retries + 1):
            try:
                async with self._sem:
                    resp = await self._client.post(
                        cfg.api_url,
                        files={cfg.file_field: ("segment.wav", wav_bytes, "audio/wav")},
                        data={cfg.lang_field: lang},
                    )
                if resp.status_code >= 500:
                    raise AsrApiError(f"ASR API HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    # 4xx — не ретраим: формат/параметры, повтор не поможет.
                    raise AsrApiError(f"ASR API rejected request (HTTP {resp.status_code}): {resp.text[:200]}")
                payload = resp.json()
                text = payload.get(cfg.text_field, "") if isinstance(payload, dict) else ""
                return AsrResult(text=(text or "").strip() if isinstance(text, str) else "", words=None)
            except (httpx.TimeoutException, httpx.TransportError, AsrApiError) as e:
                last_error = e
                if isinstance(e, AsrApiError) and "rejected" in str(e):
                    raise
                if attempt < cfg.retries:
                    delay = (2**attempt) + random.uniform(0, 0.5)
                    logger.warning("ASR attempt %d failed (%s); retry in %.1fs", attempt + 1, e, delay)
                    await asyncio.sleep(delay)
        raise AsrApiError(f"ASR API unavailable after {cfg.retries + 1} attempts: {last_error}")

    async def transcribe_batch(self, lang: str, items: list[bytes]) -> list[AsrResult]:
        return list(await asyncio.gather(*(self.transcribe(lang, b) for b in items)))

    async def aclose(self) -> None:
        await self._client.aclose()
