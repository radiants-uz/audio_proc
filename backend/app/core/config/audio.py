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
    # Батчи pyannote (по умолчанию в pyannote 32): меньше — меньше пик памяти на CPU.
    segmentation_batch_size: int = 8
    embedding_batch_size: int = 8
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

    # --- режим по кускам (по умолчанию; false = legacy «язык на спикера») ---
    chunk_lid: bool = True
    # Кусок ≥ chunk_min_sec решает сам при уверенности ≥ chunk_min_confidence:
    # 0 ошибок среди 1782 окон ≥ 2 с при ≥ 0.9 (худшая ошибка на реальных кусках 0.754).
    chunk_min_confidence: float = 0.90
    chunk_min_sec: float = 2.0
    # Кусок < chunk_min_sec решает сам только при ≥ chunk_short_confidence
    # (0.98: 80/156 коротких окон, 0 ошибок; при 0.9 — 5 ошибок).
    chunk_short_confidence: float = 0.98
    # Короткий кусок, спорящий со ВСЕМИ решёнными соседями того же спикера, решает сам
    # только при такой уверенности (корпус: верные 1.0/1.0, ошибка на русском куске 1.1 с — en 0.985).
    chunk_short_against_confidence: float = 0.995
    # Короче — не классифицируем вовсе, только наследуем.
    chunk_lid_min_sec: float = 1.0
    # Как далеко (по тому же спикеру) неуверенный кусок ищет решённого соседа.
    neighbour_max_gap_sec: float = 30.0
    # Минимум классифицированной речи спикера для мягкого приора.
    prior_min_sec: float = 2.0
    # LID по исходному сигналу (все замеры сделаны на нём), а не после шумодава.
    on_raw_audio: bool = True

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
    # Кусок, отправляемый в ASR. Сегменты диаризации до 28 с режутся на куски
    # ≤ max_chunk_sec в самых тихих точках: на дальних записях модель на длинных
    # кусках молчит, на коротких — отдаёт текст (замерено 2026-09-22: 8 с → 21/40
    # сегментов с текстом против 13/40 при 28 с).
    max_chunk_sec: float = 8.0
    # Резать куски по речевым областям silero-VAD (паддинг vad_pad_sec) вместо
    # фиксированного шага: 18/40 сегментов с текстом против 12/40 (2026-09-23).
    vad_chunks: bool = True
    vad_pad_sec: float = 0.3
    vad_min_silence_sec: float = 0.35
    # Пустой ответ → повторить кусок на других языках файла (uz/ru/en). Дорого только
    # на пустых кусках, помогает при смешанной речи и ошибке LID.
    fallback_langs: bool = True
    # Языки, которые реально обслуживает ASR-бэкенд (маршрутизация и повторы —
    # LANGID_ALLOWED ∩ ASR_LANGUAGES). Локальный stt-server без en: ["uz","ru"].
    languages: list[str] = ["uz", "ru", "en"]
    # Проверка текста ASR на «не тот язык» (textcheck.py) с повтором другим языком.
    # Безопасна по построению: повтор заменяет исходник, только если сам прошёл проверку.
    text_check: bool = True
    text_check_min_tokens: int = 3
    # Плотность текста, символов на секунду куска: верные транскрипты ≥ 7, чужие p10 = 2.3–3.0.
    text_check_min_cps: float = 4.0
    # Правило плотности только на кусках не короче этого (односложные ответы в 1–2 с нормальны).
    text_check_density_min_sec: float = 2.0
    # Максимум языков-кандидатов на повтор для одного помеченного куска.
    retry_langs_max: int = 2
    # Пустой ответ на улучшенном сигнале → повторить тот же кусок на исходном сигнале:
    # нейрошумодав иногда срезает начало реплики или тихую речь целиком (замерено:
    # DeepFilterNet терял часть русских реплик, которые сырой сигнал распознавал).
    raw_fallback: bool = True
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
    # Нейросетевой шумодав перед ASR/LID: none | deepfilter (DeepFilterNet3, Python-пакет,
    # грузится в воркере при старте; CPU ≈ 40× быстрее реального времени). Применяется
    # ко всей записи — на дальних микрофонах поднимает распознавание с нуля.
    neural: Literal["none", "deepfilter"] = "deepfilter"
    # DeepFilterNet считает файл окнами (с, перекрытие с кроссфейдом), а не целиком:
    # на 6,7-минутной записи целиком он выедал память воркера в Docker (OOM, exit 137).
    deepfilter_chunk_sec: float = 30.0
    deepfilter_overlap_sec: float = 1.0

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
