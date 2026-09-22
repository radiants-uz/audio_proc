from app.core.config import settings

from .base import DiarizationResult, Diarizer, make_exclusive


def build_diarizer(embedder=None) -> Diarizer:
    if settings.diarization.backend == "pyannote":
        from .pyannote_backend import PyannoteDiarizer

        return PyannoteDiarizer()
    from .simple_backend import SimpleDiarizer

    return SimpleDiarizer(embedder=embedder)


__all__ = ["DiarizationResult", "Diarizer", "build_diarizer", "make_exclusive"]
