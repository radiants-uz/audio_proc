"""DeepFilterNet3 (Python-пакет deepfilternet) — нейрошумодав перед ASR/LID.

Почему Python-версия, а не бинарник deep-filter: на дальней записи Python-путь
даёт 6/16 кусков с узбекским текстом (119 симв.), бинарник — 1–3 куска (20–66),
проверено 2026-09-23. Пакет требует Python ≤ 3.11 (deepfilterlib без колёс для
3.12) и старый torchaudio.backend.common — ниже небольшой shim.

Модель грузится один раз при старте воркера (ModelRegistry). CPU: 403 с аудио
≈ 10 с."""

import logging
import sys
import types

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


def _shim_torchaudio_backend() -> None:
    """deepfilternet 0.5.6 импортирует torchaudio.backend.common.AudioMetaData,
    убранный в torchaudio ≥ 2.2. Подкладываем совместимую заглушку."""
    import torchaudio

    if hasattr(torchaudio, "backend") and hasattr(torchaudio.backend, "common"):
        return
    from dataclasses import dataclass

    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int = 16
        encoding: str = "PCM_S"

    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
    torchaudio.backend = backend


class DeepFilterDenoiser:
    name = "deepfilternet3"

    def __init__(self):
        self._model = None
        self._state = None

    def load(self) -> None:
        _shim_torchaudio_backend()
        import torch
        from df import enhance, init_df

        self._model, self._state, _ = init_df(log_level="ERROR")
        self._enhance = enhance
        self._torch = torch
        logger.info("DeepFilterNet3 loaded (model sr %d)", self._state.sr())

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def denoise(self, audio16k: np.ndarray) -> np.ndarray:
        """16 kHz mono float32 → 16 kHz mono float32 (модель работает на 48 kHz).
        Файл идёт окнами ENHANCE_DEEPFILTER_CHUNK_SEC с перекрытием и кроссфейдом:
        целиком DeepFilterNet держит в памяти все промежуточные карты признаков
        и на длинной записи ронял воркер по OOM."""
        assert self.loaded, "call load() first"
        cfg = settings.enhance
        x = np.ascontiguousarray(audio16k, dtype=np.float32)
        return chunked(self._denoise_piece, x, 16000, cfg.deepfilter_chunk_sec, cfg.deepfilter_overlap_sec)

    def _denoise_piece(self, piece16k: np.ndarray) -> np.ndarray:
        import torchaudio.functional as F

        torch = self._torch
        sr = self._state.sr()
        x = torch.from_numpy(piece16k).unsqueeze(0)
        x48 = F.resample(x, 16000, sr)
        with torch.no_grad():
            y48 = self._enhance(self._model, self._state, x48)
        y = F.resample(y48, sr, 16000).squeeze(0).numpy()
        # Длина после двойного ресемплинга может отличаться на несколько отсчётов.
        n = len(piece16k)
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        return y[:n].astype(np.float32)


def chunked(fn, x: np.ndarray, sr: int, chunk_sec: float, overlap_sec: float) -> np.ndarray:
    """Применяет fn к окнам x длиной chunk_sec с перекрытием overlap_sec; на стыках —
    линейный кроссфейд. chunk_sec <= 0 или короткий файл — один вызов fn(x)."""
    n = len(x)
    step = int(chunk_sec * sr)
    ov = max(0, min(int(overlap_sec * sr), step // 2)) if step > 0 else 0
    if step <= 0 or n <= step + ov:
        return fn(x)
    out = np.zeros(n, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, ov, dtype=np.float32) if ov else None
    start = 0
    while start < n:
        end = min(n, start + step + ov)
        y = fn(x[start:end])
        if start > 0 and ov:
            # Начало окна перекрывает хвост предыдущего: плавно переходим к новому.
            out[start : start + ov] = out[start : start + ov] * (1 - ramp) + y[:ov] * ramp
            out[start + ov : end] = y[ov:]
        else:
            out[start:end] = y
        if end >= n:
            break
        start = end - ov
    return out
