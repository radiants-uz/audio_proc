"""silero-VAD — речевые области внутри сегмента для нарезки кусков ASR.

На дальних записях ASR-модель молчит, если в куске много тишины/шума:
куски, вырезанные точно по речи (с паддингом 0.3 с), дали 18/40 сегментов
с текстом против 12/40 при фиксированных 8 с (замер 2026-09-23)."""

import logging
import os

import numpy as np

from app.services.audio_processing.ingest import SAMPLE_RATE

logger = logging.getLogger(__name__)


class SileroVad:
    def __init__(self):
        self._model = None

    def load(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        # silero_vad выставляет torch.set_num_threads(1) на процесс — возвращаем.
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        self._torch = torch
        logger.info("silero-vad loaded")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def speech_regions(
        self, audio: np.ndarray, pad_sec: float = 0.3, min_silence_sec: float = 0.35, min_speech_sec: float = 0.3
    ) -> list[tuple[float, float]]:
        """Границы речи (сек, относительно начала audio)."""
        from silero_vad import get_speech_timestamps

        ts = get_speech_timestamps(
            self._torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), self._model,
            sampling_rate=SAMPLE_RATE, min_speech_duration_ms=int(min_speech_sec * 1000),
            min_silence_duration_ms=int(min_silence_sec * 1000), speech_pad_ms=int(pad_sec * 1000),
            return_seconds=True,
        )
        return [(float(t["start"]), float(t["end"])) for t in ts]
