# audio-proc — многоспикерное распознавание речи (uz / ru / en)

Сервис принимает аудиозапись совещания и возвращает транскрипт с разделением по спикерам
и языком каждого спикера, при необходимости — анализ разговора (LLM GLM-5.3 заказчика) и
отчёт PDF/DOCX. Устроен по образцу `doc-proc` и подключается к ARK тем же способом:
`X-API-Key`, `POST … → 202 {task_id}`, ARK опрашивает прогресс, результаты — presigned-ссылки на MinIO.

- Техническое задание и решения: [`STT_PIPELINE_PLAN.md`](STT_PIPELINE_PLAN.md)
- Контракт с ARK (обе стороны): [`docs/ARK_INTEGRATION.md`](docs/ARK_INTEGRATION.md)

## Пайплайн

```
audio (любой контейнер) → ffmpeg 16k mono → диаризация → сегменты (склейка/разрезание ≤ 28 с, overlap)
  → язык по спикеру (speechbrain VoxLingua107, softmax по uz/ru/en) → ASR-API заказчика по кускам
  → transcript.json → [LLM GLM-5.3 → analysis.json] → [python-docx → report.docx → LibreOffice → report.pdf]
```

| Компонент | Реализация | Переключение |
|---|---|---|
| Диаризация | `pyannote/speaker-diarization-community-1` (gated, нужен `AUDIO_HF_TOKEN`) или `simple`: silero-VAD + ECAPA-эмбеддинги + кластеризация (без токена) | `DIARIZATION_BACKEND=pyannote\|simple` |
| ASR | `POST https://stt.robot-guide.iccu.uz/api/v1/asr/transcribe` (multipart `file` + `language_code`) или `mock` для разработки | `ASR_BACKEND=remote\|mock` |
| LLM | любой OpenAI-совместимый эндпоинт (GLM-5.3 через vLLM/SGLang); сервисы заводятся в `/admin` → LLM services | строка `Service` в БД |
| PDF | LibreOffice headless в образе воркера (`WITH_LIBREOFFICE=1`) | без него — только docx + warning `pdf_failed` |

## Стек

Python 3.12, FastAPI, arq (очередь на Redis Stack), Postgres + alembic, MinIO (aioboto3), sqladmin, uv.
Одна задача на процесс воркера (модели в памяти); параллельность — числом воркеров.

## Быстрый старт (Docker)

```sh
cp .env.example .env      # поправить пароли, BACKEND_ADMIN_SECRET_KEY, порты
sh scripts/run.sh dev     # db + redis + minio + api (:8100) + worker
sh scripts/run.sh dev --scale backend-worker=2
```

- API: http://localhost:8100/docs (кнопка Authorize → `X-API-Key`), `/health` без ключа, `/v1/health/deep` с ключом.
- Админка: http://localhost:8100/admin (`BACKEND_ADMIN_USER` / `BACKEND_ADMIN_PASSWORD`). Создать LLM-сервис:
  `name=glm-5.3`, `provider=openai_compatible`, `base_url=http://<glm-host>:8000/v1`, `model_name` — как в `/v1/models` эндпоинта,
  `reasoning_effort=low`, `use_json_schema=false` (строгий json_schema через SGLang ломает ответ), `languages=["uz","ru","en"]`, `operations=["analyze","report"]`.
- MinIO: http://localhost:9201 (консоль).

Первый запуск воркера скачивает модели (~1 ГБ) в volume `hf-cache`.

## Разработка на Mac без сборки образа

```sh
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.local.yml up -d db redis minio setup-minio
cd backend && cp ../.env .env
#   в backend/.env: POSTGRES_HOST=localhost POSTGRES_PORT=5440 REDIS_HOST=localhost REDIS_PORT=6390 S3_ENDPOINT_URL=http://localhost:9200
uv sync --frozen
uv run alembic upgrade head
uv run uvicorn app.main:app --port 8100 --reload
uv run arq app.workers.arq_worker.WorkerSettings
uv run python ../scripts/fake_llm.py 8199        # фейковая GLM для проверки analyze/report
```

Нужны `ffmpeg` и (для PDF) LibreOffice: `brew install ffmpeg`, `brew install --cask libreoffice`.

CLI пайплайна без API: `uv run python -m app.services.audio_processing.run --audio file.wav [--stop-after diarize|segments|langid]`.

## API

Полное описание — `docs/ARK_INTEGRATION.md`. Кратко:

```
POST /v1/audio/transcribe   multipart file [+ expected_speakers, lid_recheck, speaker_names(json), title]      → 202 {task_id}
POST /v1/audio/analyze      + model, report_lang                                                              → 202
POST /v1/audio/report       + report_format (pdf|docx)                                                        → 202
POST /v1/audio/{document_id}/report   JSON {model, report_format, report_lang, title, speaker_names, regenerate_analysis} → 202
GET  /v1/audio/tasks/{task_id}        {status: pending|processing|complete|error, stage, current/total, current_stage/total_stages,
                                       warnings, details, processed_document, documents: {transcript, analysis, report}}
GET  /v1/audio/{document_id}          DocumentResponse + presigned url
GET  /v1/services?operation=&language=   GET /v1/languages?service=   GET /v1/operations
```

Пример:

```sh
curl -H "X-API-Key: $KEY" -F file=@meeting.m4a -F model=glm-5.3 -F report_lang=ru http://localhost:8100/v1/audio/report
curl -H "X-API-Key: $KEY" http://localhost:8100/v1/audio/tasks/<task_id>
```

## Тесты

```sh
cd backend && uv run pytest            # или sh scripts/run.sh test
```

Тестовая запись на двух спикерах (ru + en) генерируется macOS TTS: см. `eval/data/` (не в git);
`say -v Milena … ; say -v Samantha … ; ffmpeg … concat`.

## Конфигурация

Все переменные — в `.env.example` с комментариями. Ключевые:

- `AUDIO_DEVICE=cpu|cuda|mps`, `AUDIO_HF_TOKEN`, `AUDIO_MAX_DURATION_SEC=7200`
- `DIARIZATION_BACKEND`, `DIARIZATION_MIN/MAX_SPEAKERS`, `DIARIZATION_CLUSTER_THRESHOLD` (simple)
- `SEGMENTS_*` пороги склейки/разрезания, `LANGID_*` пороги языка
- `ASR_BACKEND`, `ASR_API_URL`, `ASR_CONCURRENCY`, `ASR_TIMEOUT_SEC`, `ASR_RETRIES`
- `ANALYSIS_MAX_INPUT_TOKENS=40000`, `ANALYSIS_CHUNK_TOKENS=30000` (контекст локальной GLM-5.3 — 120k)
- `ARQ_MAX_WORKERS=1`, `ARQ_JOB_TIMEOUT=3600`, `ARQ_MAX_TRIES=1`

## Известные ограничения v1

- `simple`-диаризация не помечает наложения голосов (`overlap` всегда false) и слабее pyannote на 5+ спикерах — для продакшена нужен HF-токен.
- Пословных таймкодов нет (`words: null`) — ASR-API отдаёт только текст.
- Стерео с разными людьми в каналах сводится в моно.
- Redis эфемерен: после его перезапуска незавершённые задачи для ARK «зависают» (404 на прогресс), ARK перезапускает их кнопкой Retry.
