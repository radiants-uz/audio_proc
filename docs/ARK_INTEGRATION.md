# Интеграция audio-proc ↔ ARK

Сервис `audio-proc` повторяет контракт `doc-proc`; на стороне ARK модуль `apps/audioproc`
— копия `apps/docproc` (backend) и страниц `pages/documents/processing` + `pages/admin/docproc`
(frontend). Этот документ — единственный источник правды для обеих сторон.

## 1. HTTP API audio-proc (то, что вызывает ARK)

База: `AudioProcSettings.base_url` (без `/v1`), заголовок `X-API-Key` на всё под `/v1`.

| Метод | Путь | Что |
|---|---|---|
| `GET` | `/health` | без ключа → `{"status":"ok"}` (проверка URL) |
| `GET` | `/v1/services?operation=analyze\|report&language=ru` | `[{"name":"glm-5.3"}]` — LLM-сервисы. Для `transcribe` список пуст (LLM не нужен) |
| `GET` | `/v1/languages?service=` | `[{"language_code":"uz"},{"language_code":"ru"},{"language_code":"en"}]` |
| `GET` | `/v1/operations` | `[{"operation_name":"transcribe"},{"operation_name":"analyze"},{"operation_name":"report"}]` |
| `POST` | `/v1/audio/transcribe` | multipart `file`; form: `expected_speakers?` (1–30), `lid_recheck?` (bool), `speaker_names?` (JSON-строка `{"SPEAKER_00":"Алишер"}`), `title?` → **`202 {"task_id":"…"}`** + `Location` |
| `POST` | `/v1/audio/analyze` | то же + `model` (обязателен, имя из `/v1/services`), `report_lang` (`auto\|uz\|ru\|en`, по умолчанию `auto`) |
| `POST` | `/v1/audio/report` | то же + `report_format` (`pdf\|docx`, по умолчанию `pdf`) |
| `POST` | `/v1/audio/{document_id}/report` | JSON `{"model","report_format","report_lang","title","speaker_names","regenerate_analysis"}` → `202 {"task_id"}`. `document_id` — id транскрипта (или исходника). Перестроение без повторного распознавания |
| `GET` | `/v1/audio/tasks/{task_id}` | прогресс (ниже). `404` — задача неизвестна/истекла (TTL 24 ч) |
| `GET` | `/v1/audio/{document_id}` | `DocumentResponse` с presigned `url` (24 ч) |

Ошибки: `{"detail":"…"}` (строка) или список валидации FastAPI. Коды: `401` ключ, `413` размер
(`BACKEND_MAX_FILE_SIZE_MB`, по умолчанию 500), `400` формат/длительность/параметры, `404`, `500`.

Допустимые расширения входа: wav, mp3, m4a, aac, ogg, oga, opus, flac, wma, amr, webm, mp4, mkv, mov, 3gp, mpeg, mpga.

### Прогресс `GET /v1/audio/tasks/{task_id}`

```jsonc
{
  "task_id": "uuid",
  "status": "pending | processing | complete | error",
  "created_at": "iso", "started_at": "iso", "completed_at": "iso",
  "stage": "ingest | diarization | langid | asr | assemble | analysis | report",
  "total_stages": 5, "current_stage": 3,     // transcribe=5, analyze=6, report=7, rebuild=2
  "total": 12, "current": 7,                 // внутри стадии: сегменты ASR / куски анализа
  "warnings": ["speaker_count_low"],         // опц.: no_speech, asr_mostly_empty, speakers_merged, lid_low_confidence, speaker_too_short_for_lid, analysis_failed, pdf_failed
  "details": "текст ошибки",                 // только при error
  "processed_document": { … DocumentResponse … },   // главный результат операции
  "documents": {                             // все результаты; ARK сохраняет каждый
    "transcript": { "id": 12, "name": "meeting_transcript.json", "url": "https://…", "kind": "transcript", "object_key": "…", "parent_id": 11, "operation_name": "report", "language_code": "ru", "created_at": "iso", "usage": {…} },
    "analysis":   { "id": 13, "name": "meeting_analysis.json", "url": "…", "kind": "analysis", … },   // analyze/report
    "report":     { "id": 14, "name": "meeting_report.pdf", "url": "…", "kind": "report", … }         // report
  }
}
```

`processed_document`: transcribe → transcript, analyze → analysis, report → report (файл pdf/docx;
если PDF не собрался — docx + warning `pdf_failed`). Файлы никогда не пустые.
ARK опрашивает каждые 4 с, сдаётся через 3 ч; терминальные статусы только `complete` и `error`.

### transcript.json (кратко; полная схема — STT_PIPELINE_PLAN.md §7)

```jsonc
{ "job_id": "…", "title": null,
  "audio": {"filename": "meeting.m4a", "duration_sec": 1834.2, "sample_rate": 16000},
  "speakers": [{"id": "SPEAKER_00", "name": null, "language": "uz", "lang_confidence": 0.97, "total_speech_sec": 214.3, "segments_count": 18}],
  "segments": [{"id": 0, "speaker": "SPEAKER_00", "language": "uz", "start": 12.4, "end": 18.92, "overlap": false, "overlap_with": [], "text": "…", "words": null}],
  "meta": {"pipeline_version": "1.0.0", "models": {…}, "timings_sec": {…}, "config": {…}, "warnings": []} }
```

### analysis.json (кратко; §7.1)

```jsonc
{ "report_lang": "ru", "title": "…", "summary": "…",
  "topics": [{"name","summary","segment_ids"}], "decisions": [{"text","speaker","segment_ids"}],
  "action_items": [{"text","assignee","due","segment_ids"}],
  "speakers": [{"id","role_hint","position","key_quotes":[{"segment_id","original","translated"}]}],
  "open_questions": ["…"],
  "stats": {"total_speech_sec","speakers_count","segments_count","overlap_segments","avg_segment_sec",
            "by_speaker":[{"id","name","language","speech_sec","share","segments"}], "by_language":[{"language","speech_sec","share"}]},
  "meta": {"llm","prompt_version","input_tokens","output_tokens","chunked","chunks","warnings"} }
```

## 2. ARK backend — `apps/audioproc` (копия `apps/docproc`)

- Настройки в БД (singleton `AudioProcSettings`, pk=1): `base_url`, `api_key`, `max_file_size_mb` (500),
  `max_concurrent_per_user` (3), `retention_days` (30), `default_model` (LLM для analyze/report),
  `default_report_format` (`pdf`), `default_report_lang` (`auto`), `last_checked_at/last_check_ok/last_check_error`, `updated_by/updated_at`.
- `AudioProcTask`: `created_by`, `operation` ∈ `transcribe|analyze|report`, `source_file` (StoredFile, PROTECT),
  `original_filename`, `size_bytes`, параметры `expected_speakers` (null), `model`, `report_lang`, `report_format`,
  `title`, `speaker_names` (JSON dict), статусы `pending|processing|completed|error|cancelled|expired`,
  `progress` (JSON-снимок: stage, current, total, current_stage, total_stages, warnings), `upstream_task_id`,
  `upstream_transcript_id` (document_id транскрипта в audio-proc — нужен для rebuild), `error`,
  `transcript_file`/`analysis_file`/`result_file` (StoredFile, SET_NULL; result_file — главный результат),
  `transcript_preview` (первые ~2000 символов текста реплик), `speakers_count`, `audio_duration_sec`,
  `created_at/started_at/completed_at/expires_at`.
- `client.py`: `health`, `services(operation, language)`, `languages`, `start(operation, filename, content, content_type, fields)`,
  `rebuild_report(document_id, payload)`, `progress`, `document`, `download` (с `AUDIOPROC_LOCALHOST_ALIAS`, как у docproc).
- `services.py`: `submit`, `poll` (при `complete` скачать **все** `documents.*` в бакет ARK как `audioproc/{Y/m/d}/{uuid}_{name}`, заполнить
  transcript/analysis/result_file, `transcript_preview`, `speakers_count`, `audio_duration_sec`; при `error` — `mark_error(details)`), `complete`, `notify`
  (`Notification.RouteType.AUDIOPROC`, extra `{task_id, status, operation}`), `cancel`, `retry`, `expire_results`.
- `tasks.py` (Celery): `submit_audioproc_task`, самоперепланирующийся `poll_audioproc_task` (4 с, MAX_POLL_RETRIES 10, MAX_TASK_AGE 3 ч),
  `expire_audioproc_results` (beat, 04:20).
- Права: `audioproc:use` (миграция в `auth_core`, зеркало в `setup.py PERMISSIONS`).
- Меню (миграции `menu_items` 0022–0023, зеркало в `setup.py MENU_ITEMS`): пункт `audioproc` «Audio» внутри раздела
  «Documents» (`/documents/audio`, order 3 после Processing и For signature, права `audioproc:use`). Старый маршрут `/audio`
  редиректит. В «Administration» — `audioproc_settings` «Audio processing» (`/admin-info/audioproc`, `administrator`, order 5).
- `apps.py`: `register_access_check("audioproc", …)` — владелец видит исходник и все результаты.
- `settings/base.py`: `AUDIOPROC_LOCALHOST_ALIAS` (как `DOCPROC_LOCALHOST_ALIAS`), `INSTALLED_APPS += "apps.audioproc"`,
  `urls.py`: `path("api/v1/audioproc/", include("apps.audioproc.urls"))`, `celery.py` beat.
- Аудио-MIME уже разрешены в `apps/storage/validators.py`; добавить `audio/x-wav`, `audio/amr`, `audio/opus`, `audio/3gpp`, `video/quicktime` если отсутствуют.

### REST API ARK (`/api/v1/audioproc/`, JWT, permission `audioproc:use`; admin-эндпоинты — `administrator`)

| Метод | Путь | Тело / ответ |
|---|---|---|
| `GET` | `options/` | `{configured, max_file_size_mb, max_concurrent_per_user, retention_days, extensions[], operations:[{value,label,available,needs_model}], report_formats:["pdf","docx"], report_langs:["auto","uz","ru","en"], default_model, default_report_format, default_report_lang, models:[name…], models_error}` |
| `GET` | `tasks/?status=all\|active\|completed\|error&search=&page=&page_size=` | пагинация как у docproc; `TaskSerializer` |
| `POST` | `tasks/` | `{file_id, operation, expected_speakers?, model?, report_lang?, report_format?, title?, speaker_names?}` → task |
| `GET` | `tasks/counts/` | `{all, active, completed, error}` |
| `GET` | `tasks/{id}/` | task |
| `DELETE` | `tasks/{id}/` | 204 (неактивные) |
| `POST` | `tasks/{id}/cancel/` \| `retry/` | task |
| `GET` | `tasks/{id}/transcript/` | содержимое transcript.json (объект) |
| `GET` | `tasks/{id}/analysis/` | содержимое analysis.json; 404 если нет |
| `GET` | `tasks/{id}/download/?kind=result\|transcript\|analysis\|report` | `{url, filename, mime_type, size_bytes}` |
| `POST` | `tasks/{id}/rebuild-report/` | `{report_format, report_lang, title?, speaker_names?, regenerate_analysis?, model?}` → **новая** задача `report`, привязанная к тому же исходнику (использует `upstream_transcript_id`) |
| `GET/PATCH` | `settings/` | как docproc (ключ маскируется) |
| `POST` | `settings/check/` | `{ok, services?, error?, checked_at}` |
| `GET` | `models/?refresh=1` | `{models:[{name, operations, languages}], configured, fetched_at}` |
| `GET` | `stats/?days=30` | как docproc, `by_operation` по трём операциям |

`TaskSerializer` поля: `id, operation, operation_label, original_filename, size_bytes, expected_speakers, model, report_lang, report_format, title, speaker_names, status, is_active, progress, error, source_file, transcript_file, analysis_file, result_file, transcript_preview, speakers_count, audio_duration_sec, created_at, started_at, completed_at, expires_at, duration_seconds`.

## 3. ARK frontend

- `src/api/endpoints/audioproc.ts`, `src/types/audioproc.ts` (по REST выше), `queryKeys.audioproc`, `RouteType.AudioProc = 'audioproc'` + инвалидация в `useBus.ts`.
- Маршруты: `/documents/audio` (страница, guard `['audioproc:use']`; `/audio` → редирект), `/admin-info/audioproc` (ADMIN_ONLY). `MENU_ROUTE_MAP`: `audioproc: '/documents/audio'`, `audioproc_settings: '/admin-info/audioproc'`; иконка в `Sidebar.tsx` (`AudioOutlined`, `SoundOutlined`).
- Страница `src/pages/audio/AudioProcPage.tsx` (+ `LaunchPanel.tsx`, `TaskCard.tsx`, `TranscriptDrawer.tsx`, `audioproc.css`):
  запуск — drag&drop/выбор аудио (presign-flow как в docproc), операция (транскрипт / +анализ / +отчёт), число спикеров (опц.),
  LLM-модель (для analyze/report; по умолчанию из настроек), язык отчёта, формат отчёта, название;
  список задач с фильтрами и polling 4 с пока есть активные; карточка — стадия (i18n `audioproc.stage.*`), прогресс, warnings, ошибки, кнопки:
  «Транскрипт» (drawer: участники с языком и долей, реплики по времени с цветом спикера, пометка наложений, поиск по тексту),
  «Анализ» (резюме, темы, решения, задачи, участники, открытые вопросы), «Скачать» (transcript json / analysis json / report pdf|docx),
  «Перестроить отчёт» (форма: формат, язык, имена спикеров), отмена/повтор/удаление.
- Админ-страница `src/pages/admin/audioproc/AudioProcSettingsPage.tsx`: URL, ключ, лимиты, retention, модель по умолчанию, формат и язык отчёта по умолчанию, «Проверить соединение», список моделей, статистика.
- i18n: ru / uz / en для всех строк.
