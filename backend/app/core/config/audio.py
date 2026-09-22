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
    # pyannote — основной и единственный рекомендуемый бэкенд (нужен AUDIO_HF_TOKEN);
    # simple — только для разработки без токена, включается явно.
    backend: Literal["pyannote", "simple"] = "pyannote"
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


class EnhanceSettings(BaseSettings):
    """Адаптивное улучшение звука перед ASR/LID (app/services/audio_processing/enhance.py)."""

    enabled: bool = True
    # Тихая запись: средний уровень ниже — включается speechnorm.
    quiet_db: float = -28.0
    # Шумная запись: noise floor выше — включается afftdn.
    noisy_db: float = -45.0
    denoise_nr: int = 12
    denoise_nf: int = -35
    speechnorm_expansion: float = 6.0
    # Прогонять highpass даже на чистых записях.
    always_highpass: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ENHANCE_", case_sensitive=False, extra="ignore"
    )


class SpeakersSettings(BaseSettings):
    """Слияние ложных спикеров после диаризации (app/services/audio_processing/speakers.py)."""

    enabled: bool = True
    min_speech_sec: float = 6.0
    min_share: float = 0.03
    # Порог косинусной близости для слияния пар. pyannote даёт чистые кластеры,
    # там слияние по похожести почти не нужно (0.80). simple дробит один голос на
    # обрывки с близостью 0.5–0.7 между собой, а чужой голос — 0.1–0.35
    # (замерено 2026-09-22), поэтому для него порог 0.55.
    merge_similarity: float = 0.80
    merge_similarity_simple: float = 0.55
    min_speakers: int = 1

    @property
    def similarity_for_backend(self) -> float:
        from app.core.config import settings  # noqa: PLC0415 — избегаем цикла при импорте

        return self.merge_similarity_simple if settings.diarization.backend == "simple" else self.merge_similarity

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SPEAKERS_", case_sensitive=False, extra="ignore"
    )


class ReportSettings(BaseSettings):
    libreoffice_bin: str = "soffice"
    include_full_transcript: bool = True

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="REPORT_", case_sensitive=False, extra="ignore"
    )
