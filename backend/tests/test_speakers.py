import numpy as np

from app.core.config import settings
from app.services.audio_processing.speakers import merge_speakers
from app.services.audio_processing.types import Segment

# «голоса»: A и A2 похожи, B — другой
VOICES = {
    "A": np.array([1.0, 0.0, 0.0]), "A2": np.array([0.95, 0.31, 0.0]), "B": np.array([0.0, 1.0, 0.0]),
    "N": np.array([0.6, 0.6, 0.52]),
}


def _embed_fn(segments):
    # окна → эмбеддинг спикера, которому принадлежит окно (по времени)
    def fn(audio, windows):
        out = []
        for s, _e in windows:
            seg = next(x for x in segments if abs(x.start - s) < 1e-6)
            v = VOICES[seg._voice]
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)
    return fn


def _seg(i, spk, voice, start, dur):
    s = Segment(i, spk, start, start + dur)
    s._voice = voice
    return s


def test_small_speaker_merged_into_nearest(monkeypatch):
    monkeypatch.setattr(settings.speakers, "min_speech_sec", 6.0)
    monkeypatch.setattr(settings.speakers, "merge_similarity", 0.99)
    segs = [_seg(0, "S0", "A", 0, 20), _seg(1, "S1", "B", 20, 20), _seg(2, "S2", "A2", 40, 2)]
    r = merge_speakers(_embed_fn(segs), np.zeros(16000 * 60), segs)
    assert (r.before, r.after) == (3, 2)
    assert segs[2].speaker == "S0" and r.merged[0][:2] == ("S2", "S0")


def test_similar_pair_merged(monkeypatch):
    monkeypatch.setattr(settings.speakers, "min_speech_sec", 0.0)
    monkeypatch.setattr(settings.speakers, "min_share", 0.0)
    monkeypatch.setattr(settings.diarization, "backend", "pyannote")
    monkeypatch.setattr(settings.speakers, "merge_similarity", 0.9)
    segs = [_seg(0, "S0", "A", 0, 20), _seg(1, "S1", "A2", 20, 20), _seg(2, "S2", "B", 40, 20)]
    r = merge_speakers(_embed_fn(segs), np.zeros(16000 * 60), segs)
    assert r.after == 2 and {s.speaker for s in segs} == {"S0", "S2"}


def test_expected_speakers_forces_count(monkeypatch):
    monkeypatch.setattr(settings.speakers, "min_speech_sec", 0.0)
    monkeypatch.setattr(settings.speakers, "min_share", 0.0)
    monkeypatch.setattr(settings.diarization, "backend", "pyannote")
    monkeypatch.setattr(settings.speakers, "merge_similarity", 1.01)
    segs = [_seg(0, "S0", "A", 0, 20), _seg(1, "S1", "B", 20, 20), _seg(2, "S2", "N", 40, 20), _seg(3, "S3", "A2", 60, 20)]
    r = merge_speakers(_embed_fn(segs), np.zeros(16000 * 80), segs, expected_speakers=2)
    assert r.after == 2


def test_never_below_expected(monkeypatch):
    monkeypatch.setattr(settings.speakers, "min_speech_sec", 100.0)
    segs = [_seg(0, "S0", "A", 0, 5), _seg(1, "S1", "B", 5, 5)]
    r = merge_speakers(_embed_fn(segs), np.zeros(16000 * 10), segs, expected_speakers=2)
    assert r.after == 2 and not r.merged


def test_single_speaker_untouched():
    segs = [_seg(0, "S0", "A", 0, 5)]
    r = merge_speakers(lambda a, w: np.zeros((0, 3)), np.zeros(16000 * 5), segs)
    assert (r.before, r.after) == (1, 1)


def test_simple_backend_uses_its_own_threshold(monkeypatch):
    monkeypatch.setattr(settings.speakers, "min_speech_sec", 0.0)
    monkeypatch.setattr(settings.speakers, "min_share", 0.0)
    monkeypatch.setattr(settings.speakers, "merge_similarity", 0.99)
    monkeypatch.setattr(settings.speakers, "merge_similarity_simple", 0.9)
    segs = [_seg(0, "S0", "A", 0, 20), _seg(1, "S1", "A2", 20, 20), _seg(2, "S2", "B", 40, 20)]
    monkeypatch.setattr(settings.diarization, "backend", "pyannote")
    assert merge_speakers(_embed_fn(segs), np.zeros(16000 * 60), segs).after == 3
    monkeypatch.setattr(settings.diarization, "backend", "simple")
    assert merge_speakers(_embed_fn(segs), np.zeros(16000 * 60), segs).after == 2
