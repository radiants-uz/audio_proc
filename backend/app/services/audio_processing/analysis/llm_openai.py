"""OpenAI-совместимый клиент для GLM-5.3 заказчика (vLLM / SGLang).

- JSON-ответ: по умолчанию схема вкладывается в системный промпт и просится
  response_format={"type": "json_object"} (если эндпоинт его не принимает —
  без response_format, ответ разбирается из ```json-блока). Строгий
  json_schema — только при Service.use_json_schema=True: на GLM-5.3 через
  SGLang constrained decoding даёт пустые списки и пропущенные поля
  (проверено 2026-09-20), поэтому по умолчанию он выключен;
- reasoning_effort передаётся через extra_body (vLLM/SGLang принимают его
  в теле запроса; reasoning-контент в ответе игнорируем, берём message.content);
- ретраи с экспоненциальной паузой, семафор на параллельные запросы."""

import asyncio
import json
import logging
import random
import re

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.core.config import settings
from app.models.service import Service
from app.services.audio_processing.analysis.llm_base import LlmResponse, LlmUsage

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


class LlmError(Exception):
    pass


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise LlmError("LLM returned empty content")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        return json.loads(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise LlmError(f"LLM response is not JSON: {text[:200]}")


class OpenAICompatibleClient:
    def __init__(self, service: Service):
        self.service = service
        self.model_name = service.model_name
        self._client = AsyncOpenAI(
            base_url=service.base_url,
            api_key=service.api_key or "not-needed",
            timeout=settings.analysis.timeout_sec,
            max_retries=0,
        )
        self._sem = asyncio.Semaphore(max(1, service.limit_requests or 1))

    async def complete_json(self, system: str, user: str, schema: dict, max_tokens: int) -> LlmResponse:
        if self.service.use_json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "analysis", "schema": schema, "strict": False},
            }
        else:
            system = system + "\n\nJSON schema of the answer (return exactly this structure):\n" + json.dumps(
                schema, ensure_ascii=False
            )
            response_format = {"type": "json_object"}
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        if self.service.temperature is not None:
            kwargs["temperature"] = self.service.temperature
        extra_body = {}
        if self.service.reasoning_effort:
            extra_body["reasoning_effort"] = self.service.reasoning_effort
            # SGLang / некоторые vLLM-сборки ждут chat_template_kwargs
            extra_body["chat_template_kwargs"] = {"reasoning_effort": self.service.reasoning_effort}
        if extra_body:
            kwargs["extra_body"] = extra_body

        last: Exception | None = None
        for attempt in range(settings.analysis.retries + 1):
            try:
                async with self._sem:
                    resp = await self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                data = _extract_json(content)
                u = resp.usage
                usage = LlmUsage(
                    input_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(u, "completion_tokens", 0) or 0),
                    calls=1,
                )
                return LlmResponse(data=data, usage=usage)
            except APIStatusError as e:
                last = e
                # response_format не поддерживается — падаем на промпт-режим один раз.
                if e.status_code in (400, 422) and "response_format" in kwargs:
                    logger.warning("LLM rejected response_format (%s); retrying without it", e.message[:200])
                    kwargs.pop("response_format")
                    continue
                if e.status_code < 500 and e.status_code != 429:
                    raise LlmError(f"LLM HTTP {e.status_code}: {e.message[:300]}") from e
            except (APIConnectionError, APITimeoutError, json.JSONDecodeError, LlmError) as e:
                last = e
            if attempt < settings.analysis.retries:
                delay = 2**attempt + random.uniform(0, 1)
                logger.warning("LLM attempt %d failed (%s); retry in %.1fs", attempt + 1, last, delay)
                await asyncio.sleep(delay)
        raise LlmError(f"LLM unavailable after {settings.analysis.retries + 1} attempts: {last}") from last
