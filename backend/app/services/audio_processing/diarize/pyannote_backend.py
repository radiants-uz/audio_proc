"""§5.2: pyannote.audio (gated-модель, нужен HF_TOKEN)."""

import logging
from pathlib import Path

from app.core.config import settings
from app.services.audio_processing.diarize.base import DiarizationResult, make_exclusive
from app.services.audio_processing.types import RawTurn

logger = logging.getLogger(__name__)


class PyannoteDiarizer:
    def __init__(self):
        self.model_name = settings.diarization.model
        self._pipeline = None

    def load(self) -> None:
        import torch
        from pyannote.audio import Pipeline

        token = settings.audio.hf_token.get_secret_value() or None
        if not token:
            raise RuntimeError(
                "DIARIZATION_BACKEND=pyannote requires AUDIO_HF_TOKEN: create a free Hugging Face "
                f"account, accept the terms at https://huggingface.co/{self.model_name}, then put a "
                "read token into AUDIO_HF_TOKEN (or set DIARIZATION_BACKEND=simple for development)"
            )
        self._pipeline = Pipeline.from_pretrained(self.model_name, token=token)
        # Меньшие батчи — меньший пик памяти на CPU (файл 6,7 мин ронял воркер по OOM).
        for attr in ("segmentation_batch_size", "embedding_batch_size"):
            if hasattr(self._pipeline, attr):
                setattr(self._pipeline, attr, getattr(settings.diarization, attr))
        self._pipeline.to(torch.device(settings.audio.device))
        self._torch = torch
        logger.info("pyannote pipeline %s loaded on %s", self.model_name, settings.audio.device)

    def diarize(self, wav_path: Path, min_speakers: int | None, max_speakers: int | None) -> DiarizationResult:
        assert self._pipeline is not None, "call load() first"
        kwargs = {}
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers
        # Передаём waveform, а не путь: pyannote 4 декодирует файлы через torchcodec,
        # который требует совместимых библиотек ffmpeg (на Mac с ffmpeg 8 не грузится).
        # У нас уже есть 16 kHz mono wav — читаем soundfile и отдаём тензор.
        import soundfile as sf

        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = self._torch.from_numpy(data.T.copy())  # (channels, samples)
        output = self._pipeline({"waveform": waveform, "sample_rate": int(sr)}, **kwargs)

        # pyannote 4.x возвращает объект с .speaker_diarization / .exclusive_speaker_diarization,
        # 3.x — Annotation напрямую.
        annotation = getattr(output, "speaker_diarization", output)
        exclusive = getattr(output, "exclusive_speaker_diarization", None)

        turns = [
            RawTurn(start=float(seg.start), end=float(seg.end), speaker=str(label))
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]
        if exclusive is not None:
            exclusive_turns = [
                RawTurn(start=float(seg.start), end=float(seg.end), speaker=str(label))
                for seg, _, label in exclusive.itertracks(yield_label=True)
            ]
        else:
            exclusive_turns = make_exclusive(turns)
        return DiarizationResult(turns=turns, exclusive_turns=exclusive_turns, model_name=self.model_name)
