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
| Язык | VoxLingua107 (speechbrain) **на каждый кусок ASR** с наследованием от соседей того же спикера; текст ASR проверяется символьными n-граммами и при «не том языке» повторяется другой моделью (`textcheck.py`) | `LANGID_CHUNK_*`, `ASR_TEXT_CHECK` |
| Диаризация | **`pyannote/speaker-diarization-community-1`** (по умолчанию; gated, нужен `AUDIO_HF_TOKEN`). `simple` (silero-VAD + ECAPA + кластеризация) — только для разработки без токена | `DIARIZATION_BACKEND=pyannote\|simple` |
| ASR | `POST https://stt.robot-guide.iccu.uz/api/v1/asr/transcribe` (multipart `file` + `language_code`) или `mock` для разработки | `ASR_BACKEND=remote\|mock` |
| LLM | любой OpenAI-совместимый эндпоинт (GLM-5.3 через vLLM/SGLang); сервисы заводятся в `/admin` → LLM services | строка `Service` в БД |
| Улучшение звука | перед ASR/LID: нейрошумодав DeepFilterNet3 (Python-пакет, грузится в воркере) + адаптивно тихо → `speechnorm`, шумно → `afftdn`; диаризация по исходнику | `ENHANCE_NEURAL=deepfilter\|none`, `ENHANCE_*` |
| Спикеры | слияние ложных спикеров по ECAPA-эмбеддингам (мелкие, похожие, `expected_speakers`) | `SPEAKERS_*` |
| PDF | LibreOffice headless в образе воркера (`WITH_LIBREOFFICE=1`) | без него — только docx + warning `pdf_failed` |

## Стек

Python 3.11 (deepfilternet не имеет колёс для 3.12+), FastAPI, arq (очередь на Redis Stack), Postgres + alembic, MinIO (aioboto3), sqladmin, uv.
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

## Prod (сервер)

```sh
sh scripts/run.sh up   # docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

- Снаружи стоит хостовый nginx сервера (:80, конфиг `~/nginx-host/` на сервере): он проксирует
  `/audio-proc/` → nginx проекта на `127.0.0.1:${NGINX_HTTP_HOST_PORT}` (8081). Наружу порт проекта не публикуется.
- Префикс срезается на хосте, приложение получает его через `UVICORN_ROOT_PATH` (= `API_ROOT_PATH` из `.env`):
  API — `http://<host>/audio-proc/docs`, админка — `http://<host>/audio-proc/admin`. Свой домен с корня — `API_ROOT_PATH=`.
- Хранилище — внешнее S3 (`S3_ENDPOINT_URL`, `S3_EXTERNAL_URL`, ключи, бакет в `.env`); MinIO в prod-compose
  не входит. Нужен локальный MinIO — `sh scripts/run.sh --with-minio up` (добавляет `docker-compose.minio.yml`).
- TLS терминируется на хосте (certbot), контейнерный nginx (`.docker/nginx/`) слушает только HTTP.

## Конфигурация

Все переменные — в `.env.example` с комментариями. Ключевые:

- `AUDIO_DEVICE=cpu|cuda|mps`, `AUDIO_HF_TOKEN`, `AUDIO_MAX_DURATION_SEC=7200`
- `DIARIZATION_BACKEND`, `DIARIZATION_MIN/MAX_SPEAKERS`, `DIARIZATION_CLUSTER_THRESHOLD` (simple)
- `SEGMENTS_*` пороги склейки/разрезания, `LANGID_*` пороги языка
- `ASR_BACKEND`, `ASR_API_URL`, `ASR_CONCURRENCY`, `ASR_TIMEOUT_SEC`, `ASR_RETRIES`
- `ANALYSIS_MAX_INPUT_TOKENS=40000`, `ANALYSIS_CHUNK_TOKENS=30000` (контекст локальной GLM-5.3 — 120k)
- `ARQ_MAX_WORKERS=1`, `ARQ_JOB_TIMEOUT=3600`, `ARQ_MAX_TRIES=1`

## Нейросетевая чистка звука (2026-09-23)

Кандидаты сравнивались на 16 речевых кусках (3–12 с) дальней записи совещания; метрика — сколько текста возвращает ASR
(uz-модель NeMo и ru-модель GigaAM) после обработки + `speechnorm`. Сырой сигнал: uz 0 кусков, ru 270 симв. (обрывки).

| Модель | uz-текст (кусков/16) | ru-текст, симв. | CPU на 5-с кусок |
|---|---|---|---|
| **DeepFilterNet3 (Python)** | **6/16, 119 симв.** | 99 | 0.3 с |
| DeepFilterNet3 (бинарник deep-filter) | 1–3/16, 20–66 | 1–11 | 0.3 с |
| ClearerVoice MossFormerGAN_SE_16K | 5/16, 87 | 312 | 29 с |
| ClearerVoice MossFormer2_SE_48K | 0/16, 4 | 193 | 2 с |
| ClearerVoice FRCRN_SE_16K | 0/16, 3 | 201 | 47 с |
| Resemble Enhance (denoise) | 0/16 | ≈ сырой | 5 с |
| Resemble Enhance (enhance, диффузия) | 13/16, 337 симв. — **выдуманная речь** | 28 | 38 с |
| ffmpeg speechnorm | 0/16 | 243 | 0.1 с |

Выбран DeepFilterNet3 через Python-пакет: единственный, кто даёт узбекский текст при приемлемой цене на CPU
(403 с записи ≈ 20 с). MossFormerGAN сопоставим по узбекскому, но в 100 раз медленнее. Resemble Enhance в режиме
enhance даёт больше всего «текста», но это генеративная модель: она дорисовывает речь, и ASR распознаёт правдоподобные,
но выдуманные фразы («Dubaydagi Sarsang London», «U Britaniya uchun shunday joy») — для протоколов совещаний неприемлемо.
Метрика «сколько символов» для генеративных улучшателей не годится, нужна сверка с тем, что реально сказано. Шумодав иногда срезает тихую
речь, поэтому пустой кусок повторно распознаётся на исходном сигнале (`ASR_RAW_FALLBACK`), а затем на других языках
(`ASR_FALLBACK_LANGS`). Куски для ASR режутся по речи silero-VAD (`ASR_VAD_CHUNKS`, паддинг 0.3 с) и не длиннее `ASR_MAX_CHUNK_SEC` (8 с):
на длинных кусках и кусках с тишиной модель молчит (VAD-нарезка: 18/40 сегментов с текстом против 12/40 фиксированными 8 с).

## Определение языка (2026-09-24)

Язык решается на каждый кусок ASR, а не на спикера: один человек может говорить по-русски, потом по-узбекски. На 158 реальных
кусках (12 смешанных uz/ru/en файлов + 3 записи) схема «язык на спикера» давала 71 % кусков верно, по кускам — 158/158;
после ревью и правок (2026-09-24) — 164/164 кусков на 14 файлах с ASR uz/ru/en.
Пороги: кусок ≥ 2 с решает сам при уверенности ≥ 0.90, короче 2 с — при ≥ 0.98, иначе наследует язык ближайшего решённого
куска того же спикера (при разногласии соседей решает аудио самого куска). Страховка по тексту: русская речь через узбекскую
модель возвращается транслитерацией («Zagruziti zapes Strechi…»), такой текст ловят символьные 5-граммные модели (0 ошибок на
370 парах) и кусок повторяется нужной моделью; повтор по пустому ответу проходит ту же проверку. Отвергнутый бэкендом кусок
(4xx) не валит задачу и не выключает язык. Подробности и все пороги — STT_PIPELINE_PLAN.md §5.4, §5.6.

## Замеры диаризации (2026-09-22, Apple M4, CPU)

| Запись | pyannote community-1 | simple |
|---|---|---|
| Совещание, дальний микрофон, 403 с | 2 спикера, 237 с | 15 «спикеров», 10 с |
| Диалог-сценка, 172 с | 2 спикера, 108 с | 14 «спикеров», 4 с |
| Синтез, 2 голоса, 33 с | 2, 22 с | 2, 1 с |

pyannote на CPU работает примерно в 0.6× реального времени (час записи → ~35 мин) и требует 2–3 ГБ RAM
сверх загруженных моделей. `simple` быстрый, но на реальных записях дробит голоса на обрывки — для него
обязательна пост-обработка `SPEAKERS_*`, и всё равно он только для разработки.

## Известные ограничения v1

- `simple`-диаризация не помечает наложения голосов (`overlap` всегда false) и слабее pyannote на 5+ спикерах — для продакшена нужен HF-токен.
- Пословных таймкодов нет (`words: null`) — ASR-API отдаёт только текст.
- Стерео с разными людьми в каналах сводится в моно.
- Redis эфемерен: после его перезапуска незавершённые задачи для ARK «зависают» (404 на прогресс), ARK перезапускает их кнопкой Retry.
