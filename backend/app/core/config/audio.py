"""Настройки аудио-пайплайна (§5, §9 плана). Каждая группа — свой env-префикс."""

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioSettings(BaseSettings):
    max_duration_sec: float = 7200
    min_duration_sec: float = 1.0
    tmp_dir: str = "/tmp/audio-proc"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    hf_token: SecretStr = SecretStr("")
    keep_wav: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AUDIO_", case_sensitive=False, extra="ignore"
    )


class DiarizationSettings(BaseSettings):
    backend: Literal["pyannote", "simple"] = "simple"
    model: str = "pyannote/speaker-diarization-community-1"
    min_speakers: int = 1
    max_speakers: int = 15
    # simple-бэкенд (без HF-токена): VAD + эмбеддинги + агломеративная кластеризация.
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    cluster_threshold: float = 0.7
    window_sec: float = 1.5
    step_sec: float = 0.75

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="DIARIZATION_", case_sensitive=False, extra="ignore"
    )


class SegmentsSettings(BaseSettings):
    merge_gap_sec: float = 0.5
    min_segment_sec: float = 0.4
    max_segment_sec: float = 28.0
    overlap_min_sec: float = 0.3
    pad_sec: float = 0.15

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SEGMENTS_", case_sensitive=False, extra="ignore"
    )


class LangIdSettings(BaseSettings):
    model: str = "speechbrain/lang-id-voxlingua107-ecapa"
    allowed: list[str] = ["uz", "ru", "en"]
    audio_sec: float = 20.0
    min_audio_sec: float = 2.0
    min_confidence: float = 0.6
    per_segment_recheck: bool = False
    recheck_min_segment_sec: float = 4.0
    recheck_min_confidence: float = 0.9

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LANGID_", case_sensitive=False, extra="ignore"
    )


class AsrSettings(BaseSettings):
    backend: Literal["remote", "mock"] = "remote"
    api_url: str = "https://stt.robot-guide.iccu.uz/api/v1/asr/transcribe"
    file_field: str = "file"
    lang_field: str = "language_code"
    text_field: str = "transcription"
    auth_header: str | None = None
    auth_token: SecretStr = SecretStr("")
    concurrency: int = 4
    timeout_sec: float = 60.0
    retries: int = 3
    # Доля реплик с пустым текстом, начиная с которой ставится warning asr_mostly_empty.
    empty_warning_ratio: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ASR_", case_sensitive=False, extra="ignore"
    )


class AnalysisSettings(BaseSettings):
    # Контекст локально поднятой GLM-5.3 — 120k токенов: 40k на вход,
    # остальное — промпт, reasoning и вывод.
    max_input_tokens: int = 40000
    chunk_tokens: int = 30000
    max_output_tokens: int = 8000
    timeout_sec: float = 300.0
    retries: int = 3
    default_report_lang: str = "auto"

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ANALYSIS_", case_sensitive=False, extra="ignore"
    )


class ReportSettings(BaseSettings):
    libreoffice_bin: str = "soffice"
    include_full_transcript: bool = True

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="REPORT_", case_sensitive=False, extra="ignore"
    )
