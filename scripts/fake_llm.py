"""Фейковый OpenAI-совместимый эндпоинт для разработки без GLM-5.3.

Отвечает на POST /v1/chat/completions валидным analysis-JSON (схема §7.1),
поддерживает response_format json_schema и reasoning_effort (игнорирует).
Запуск:  uv run --project backend python scripts/fake_llm.py  (порт 8199)
Сервис в /admin:  name=glm-fake, base_url=http://host:8199/v1, model_name=glm-5.3
"""

import json
import re
import sys
import time

from fastapi import FastAPI, Request

app = FastAPI(title="fake GLM")

_ID_RE = re.compile(r"#(\d+)\s")


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "glm-5.3", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    user = body["messages"][-1]["content"]
    ids = [int(x) for x in _ID_RE.findall(user)][:50]
    lang = "ru"
    for cand, marker in (("uz", "Siz — yig'ilishlar"), ("en", "You are a meeting analyst")):
        if marker in body["messages"][0]["content"]:
            lang = cand
    summary = {
        "ru": "Команда обсудила сроки второго этапа проекта. Решено стартовать в понедельник, график пересмотреть на следующей неделе.",
        "uz": "Jamoa loyihaning ikkinchi bosqichi muddatlarini muhokama qildi. Dushanba kuni boshlashga qaror qilindi.",
        "en": "The team discussed the timeline of the second phase. It was decided to start on Monday and revisit the schedule next week.",
    }[lang]
    is_reduce = "part 1" in user
    data = {
        "title": {"ru": "Планёрка по второму этапу", "uz": "Ikkinchi bosqich bo'yicha yig'ilish", "en": "Phase two kick-off"}[lang],
        "summary": summary + (" (reduce)" if is_reduce else ""),
        "topics": [{"name": {"ru": "Сроки", "uz": "Muddatlar", "en": "Timeline"}[lang], "summary": summary, "segment_ids": ids[:3]}],
        "decisions": [{"text": {"ru": "Старт в понедельник", "uz": "Dushanba kuni boshlash", "en": "Start on Monday"}[lang], "speaker": "SPEAKER_00", "segment_ids": ids[2:3] or ids[:1]}],
        "action_items": [{"text": {"ru": "Подготовить график", "uz": "Jadval tayyorlash", "en": "Prepare the schedule"}[lang], "assignee": "SPEAKER_01", "due": {"ru": "к пятнице", "uz": "juma kunigacha", "en": "by Friday"}[lang], "segment_ids": ids[3:4] or ids[:1]}],
        "speakers": [
            {"id": "SPEAKER_00", "role_hint": {"ru": "ведущий", "uz": "boshlovchi", "en": "facilitator"}[lang], "position": summary[:60], "key_quotes": [{"segment_id": ids[0] if ids else 0, "original": "Коллеги, давайте начнём.", "translated": "Colleagues, let's begin."}]},
            {"id": "SPEAKER_01", "role_hint": None, "position": summary[:40], "key_quotes": []},
        ],
        "open_questions": [{"ru": "Кто отвечает за бюджет?", "uz": "Byudjet uchun kim javobgar?", "en": "Who owns the budget?"}[lang]],
    }
    content = json.dumps(data, ensure_ascii=False)
    prompt_tokens = sum(len(m["content"]) for m in body["messages"]) // 3
    return {
        "id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()), "model": body.get("model", "glm-5.3"),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content, "reasoning_content": "thinking…"}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(content) // 3, "total_tokens": prompt_tokens + len(content) // 3},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8199, log_level="warning")
