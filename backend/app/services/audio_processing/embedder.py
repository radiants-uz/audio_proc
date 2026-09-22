"""ECAPA-эмбеддинги голоса (speechbrain, не gated). Общий для simple-диаризации
и для слияния ложных спикеров после любого бэкенда."""

import logging
import os

import numpy as np

from app.core.config import settings
from app.services.audio_processing.ingest import SAMPLE_RATE

logger = logging.getLogger(__name__)

# На CPU батч из 32 окон считается в десятки раз дольше, чем 4 батча по 8
# (замерено на Apple M4: 18 с против 0.4 с) — держим батч маленьким.
EMBED_BATCH = 8


class SpeakerEmbedder:
    def __init__(self):
        self.model_name = settings.diarization.embedding_model
        self._encoder = None

    def load(self) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self._encoder = EncoderClassifier.from_hparams(
            source=self.model_name,
            run_opts={"device": settings.audio.device if settings.audio.device != "mps" else "cpu"},
        )
        self._torch = torch
        # silero_vad (если грузился раньше) выставляет torch.set_num_threads(1) на процесс.
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        logger.info("speaker embedder %s loaded", self.model_name)

    def embed(self, audio: np.ndarray, windows: list[tuple[float, float]], max_sec: float = 1.5) -> np.ndarray:
        """L2-нормированные эмбеддинги окон (start, end) в секундах. Окна длиннее
        max_sec режутся на подокна, эмбеддинг окна — среднее подокон."""
        assert self._encoder is not None, "call load() first"
        pieces: list[tuple[int, np.ndarray]] = []  # (индекс окна, сигнал)
        for wi, (s, e) in enumerate(windows):
            t = s
            while t < e - 0.05:
                a, b = int(t * SAMPLE_RATE), int(min(e, t + max_sec) * SAMPLE_RATE)
                x = audio[a:b]
                if len(x) < int(0.3 * SAMPLE_RATE):
                    x = np.pad(x, (0, int(0.3 * SAMPLE_RATE) - len(x)))
                pieces.append((wi, x))
                t += max_sec
        if not pieces:
            return np.zeros((0, 192), dtype=np.float32)
        maxlen = int(max_sec * SAMPLE_RATE)
        embs = np.zeros((len(pieces), 192), dtype=np.float32)
        for i in range(0, len(pieces), EMBED_BATCH):
            batch = pieces[i : i + EMBED_BATCH]
            padded = np.zeros((len(batch), maxlen), dtype=np.float32)
            lens = np.zeros(len(batch), dtype=np.float32)
            for j, (_, x) in enumerate(batch):
                padded[j, : len(x)] = x[:maxlen]
                lens[j] = min(len(x), maxlen) / maxlen
            with self._torch.no_grad():
                e = self._encoder.encode_batch(self._torch.from_numpy(padded), self._torch.from_numpy(lens))
            embs[i : i + len(batch)] = e.squeeze(1).cpu().numpy()
        out = np.zeros((len(windows), 192), dtype=np.float32)
        counts = np.zeros(len(windows))
        for (wi, _), e in zip(pieces, embs, strict=True):
            out[wi] += e
            counts[wi] += 1
        out = out / np.maximum(counts, 1)[:, None]
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
