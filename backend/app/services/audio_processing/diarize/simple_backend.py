"""Диаризация без HF-токена: silero-VAD → окна речи → ECAPA-эмбеддинги
(speechbrain, не gated) → агломеративная кластеризация по косинусному
расстоянию. Наложения не детектирует (turns == exclusive_turns).
Годится для разработки и как запасной вариант; для продакшена — pyannote."""

import logging
import os
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.services.audio_processing.diarize.base import DiarizationResult
from app.services.audio_processing.ingest import SAMPLE_RATE, load_wav
from app.services.audio_processing.types import RawTurn

logger = logging.getLogger(__name__)



class SimpleDiarizer:
    def __init__(self, embedder=None):
        self.model_name = f"simple(silero-vad + {settings.diarization.embedding_model})"
        self._vad = None
        self._embedder = embedder

    def load(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        from app.services.audio_processing.embedder import SpeakerEmbedder

        self._vad = load_silero_vad()
        if self._embedder is None:
            self._embedder = SpeakerEmbedder()
            self._embedder.load()
        # silero_vad выставляет torch.set_num_threads(1) на весь процесс — возвращаем.
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        self._torch = torch
        logger.info("simple diarizer loaded (%s)", self.model_name)

    # ---- helpers -------------------------------------------------------
    def _speech_regions(self, audio: np.ndarray) -> list[tuple[float, float]]:
        from silero_vad import get_speech_timestamps

        ts = get_speech_timestamps(
            self._torch.from_numpy(audio), self._vad, sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=250, min_silence_duration_ms=300, speech_pad_ms=100,
            return_seconds=True,
        )
        return [(float(t["start"]), float(t["end"])) for t in ts]

    def _windows(self, regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
        win, step = settings.diarization.window_sec, settings.diarization.step_sec
        out = []
        for s, e in regions:
            if e - s <= win:
                out.append((s, e))
                continue
            t = s
            while t + win <= e + 1e-6:
                out.append((t, t + win))
                t += step
            if out[-1][1] < e - 0.3:
                out.append((max(s, e - win), e))
        return out

    def _embed(self, audio: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
        return self._embedder.embed(audio, windows, max_sec=settings.diarization.window_sec)

    def _cluster(self, embs: np.ndarray, min_speakers: int, max_speakers: int) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        n = len(embs)
        if n == 0:
            return np.zeros(0, dtype=int)
        if n == 1 or max_speakers == 1:
            return np.zeros(n, dtype=int)
        min_speakers = max(1, min(min_speakers, n))
        max_speakers = max(min_speakers, min(max_speakers, n))
        if min_speakers == max_speakers:
            model = AgglomerativeClustering(n_clusters=min_speakers, metric="cosine", linkage="average")
            return model.fit_predict(embs)
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=settings.diarization.cluster_threshold,
            metric="cosine", linkage="average",
        )
        labels = model.fit_predict(embs)
        k = len(set(labels))
        if k < min_speakers or k > max_speakers:
            k = min(max(k, min_speakers), max_speakers)
            model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            labels = model.fit_predict(embs)
        return labels

    # ---- API -----------------------------------------------------------
    def diarize(self, wav_path: Path, min_speakers: int | None, max_speakers: int | None) -> DiarizationResult:
        assert self._vad is not None, "call load() first"
        audio = load_wav(wav_path)
        regions = self._speech_regions(audio)
        windows = self._windows(regions)
        if not windows:
            return DiarizationResult(turns=[], exclusive_turns=[], model_name=self.model_name)

        embs = self._embed(audio, windows)
        labels = self._cluster(
            embs, min_speakers or settings.diarization.min_speakers,
            max_speakers or settings.diarization.max_speakers,
        )

        # Окна перекрываются: голос за момент времени — метка окна с ближайшим центром.
        # Собираем turns: соседние окна одной метки склеиваем.
        turns: list[RawTurn] = []
        for (s, e), lab in sorted(zip(windows, labels, strict=False), key=lambda x: x[0][0]):
            spk = f"SPK{int(lab)}"
            if turns and turns[-1].speaker == spk and s <= turns[-1].end + 0.05:
                turns[-1].end = max(turns[-1].end, e)
            elif turns and s < turns[-1].end:
                # окно другого спикера начинается внутри предыдущего — режем по середине перекрытия
                mid = (s + turns[-1].end) / 2
                turns[-1].end = mid
                turns.append(RawTurn(start=mid, end=e, speaker=spk))
            else:
                turns.append(RawTurn(start=s, end=e, speaker=spk))
        turns = [t for t in turns if t.duration > 0.05]
        logger.info("simple diarizer: %d speech regions, %d windows, %d speakers", len(regions), len(windows), len(set(labels)))
        return DiarizationResult(turns=turns, exclusive_turns=list(turns), model_name=self.model_name)
