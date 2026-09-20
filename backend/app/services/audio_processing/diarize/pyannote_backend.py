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
                "DIARIZATION_BACKEND=pyannote requires AUDIO_HF_TOKEN (gated model on Hugging Face)"
            )
        self._pipeline = Pipeline.from_pretrained(self.model_name, token=token)
        self._pipeline.to(torch.device(settings.audio.device))
        logger.info("pyannote pipeline %s loaded on %s", self.model_name, settings.audio.device)

    def diarize(self, wav_path: Path, min_speakers: int | None, max_speakers: int | None) -> DiarizationResult:
        assert self._pipeline is not None, "call load() first"
        kwargs = {}
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers
        output = self._pipeline(str(wav_path), **kwargs)

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
