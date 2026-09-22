"""Реестр моделей воркера: грузятся один раз при старте процесса."""

import logging
import time

from app.core.config import settings
from app.services.audio_processing.denoise import DeepFilterDenoiser
from app.services.audio_processing.diarize import Diarizer, build_diarizer
from app.services.audio_processing.embedder import SpeakerEmbedder
from app.services.audio_processing.langid import LanguageIdentifier
from app.services.audio_processing.vad import SileroVad

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        # ECAPA-эмбеддер общий: simple-диаризация и слияние ложных спикеров.
        self.embedder = SpeakerEmbedder()
        self.diarizer: Diarizer = build_diarizer(embedder=self.embedder)
        self.langid = LanguageIdentifier()
        self.denoiser = DeepFilterDenoiser()
        self.vad = SileroVad()
        self.loaded = False

    def load_all(self) -> None:
        t0 = time.perf_counter()
        self.embedder.load()
        self.diarizer.load()
        self.langid.load()
        if settings.enhance.neural == "deepfilter":
            self.denoiser.load()
        if settings.asr.vad_chunks:
            self.vad.load()
        self.loaded = True
        logger.info("models loaded in %.1fs", time.perf_counter() - t0)
