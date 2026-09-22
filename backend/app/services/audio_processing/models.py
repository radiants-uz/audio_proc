"""Реестр моделей воркера: грузятся один раз при старте процесса."""

import logging
import time

from app.services.audio_processing.diarize import Diarizer, build_diarizer
from app.services.audio_processing.embedder import SpeakerEmbedder
from app.services.audio_processing.langid import LanguageIdentifier

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        # ECAPA-эмбеддер общий: simple-диаризация и слияние ложных спикеров.
        self.embedder = SpeakerEmbedder()
        self.diarizer: Diarizer = build_diarizer(embedder=self.embedder)
        self.langid = LanguageIdentifier()
        self.loaded = False

    def load_all(self) -> None:
        t0 = time.perf_counter()
        self.embedder.load()
        self.diarizer.load()
        self.langid.load()
        self.loaded = True
        logger.info("models loaded in %.1fs", time.perf_counter() - t0)
