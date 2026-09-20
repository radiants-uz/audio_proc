# syntax=docker/dockerfile:1

# =====================================================================
# base — uv + Python 3.12 (torch/pyannote/speechbrain ещё не собраны под
# 3.14, это единственное отличие от doc-proc). venv лежит в /opt/venv
# (ВНЕ /app), поэтому bind-mount поверх /app в dev его не перекрывает.
# =====================================================================
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# uid/gid 1000 — чтобы updateRemoteUserUID девконтейнера совпал с юзером хоста.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser \
    && install -d -o appuser -g appuser /opt/venv /app /tmp/audio-proc

# ffmpeg — ingest любого контейнера в wav 16k mono; libsndfile — soundfile;
# шрифты Noto/DejaVu — кириллица и узбекская латиница (oʻ gʻ) в PDF-отчётах.
# LibreOffice (~400 МБ) нужен только для docx → pdf; --build-arg WITH_LIBREOFFICE=0
# отключает его, тогда отчёт отдаётся только в docx.
ARG WITH_LIBREOFFICE=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libgomp1 curl fonts-dejavu-core fonts-noto-core \
    && if [ "$WITH_LIBREOFFICE" = "1" ]; then \
        apt-get install -y --no-install-recommends libreoffice-writer fonts-liberation; \
    fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =====================================================================
# deps — только сторонние зависимости, кэшируются по lock-файлам.
# =====================================================================
FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# =====================================================================
# dev — цель для docker-compose.dev.yml: dev-зависимости, исходники
# bind-монтируются.
# =====================================================================
FROM deps AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
RUN chown -R appuser:appuser /opt/venv
USER appuser

# =====================================================================
# builder — исходники поверх готовых зависимостей для прод-образа.
# =====================================================================
FROM deps AS builder
COPY . .

# =====================================================================
# prod — тонкий рантайм, без uv: только venv + приложение, non-root.
# =====================================================================
FROM python:3.12-slim-trixie AS prod

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ARG WITH_LIBREOFFICE=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libgomp1 curl fonts-dejavu-core fonts-noto-core \
    && if [ "$WITH_LIBREOFFICE" = "1" ]; then \
        apt-get install -y --no-install-recommends libreoffice-writer fonts-liberation; \
    fi \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser \
    && install -d -o appuser -g appuser /tmp/audio-proc

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
