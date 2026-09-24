import numpy as np
import pytest

from app.core.config import settings
from app.services.audio_processing.langid import LangResult, label_chunks, speaker_languages
from app.services.audio_processing.types import Chunk


class FakeLid:
    """probs по (start, end) куска; allowed как в проекте."""

    allowed = ["uz", "ru", "en"]

    def __init__(self, table):
        self.table = table
        self.calls = []

    def classify(self, audio):
        # аудио — срез; ищем по длине (тест задаёт уникальные длительности)
        n = len(audio) / 16000
        self.calls.append(round(n, 2))
        for (a, b), probs in self.table.items():
            if abs((b - a) - n) < 1e-3:
                best = max(probs, key=probs.get)
                return LangResult(language=best, confidence=probs[best], probs=probs)
        raise AssertionError(f"no fake probs for a {n:.2f}s chunk")


AUDIO = np.zeros(16000 * 60, dtype="float32")


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(settings.langid, "chunk_min_confidence", 0.9)
    monkeypatch.setattr(settings.langid, "chunk_min_sec", 2.0)
    monkeypatch.setattr(settings.langid, "chunk_short_confidence", 0.98)
    monkeypatch.setattr(settings.langid, "chunk_short_against_confidence", 0.995)
    monkeypatch.setattr(settings.langid, "chunk_lid_min_sec", 1.0)
    monkeypatch.setattr(settings.langid, "neighbour_max_gap_sec", 30.0)
    monkeypatch.setattr(settings.langid, "prior_min_sec", 2.0)


def _ch(seg, idx, a, b, spk="S0"):
    return Chunk(seg_id=seg, idx=idx, start=a, end=b, speaker=spk)


def test_code_switching_speaker_gets_two_languages():
    """rec16k: одна и та же персона, русская фраза 1.9–7.2 и узбекская 11.1–16.2."""
    chunks = [_ch(0, 0, 1.9, 7.2), _ch(1, 0, 11.1, 16.2)]
    lid = FakeLid({(1.9, 7.2): {"uz": 0.0001, "ru": 0.9999, "en": 0.0}, (11.1, 16.2): {"uz": 1.0, "ru": 0.0, "en": 0.0}})
    warnings = []
    label_chunks(lid, AUDIO, chunks, warnings)
    assert [(c.language, c.lang_source) for c in chunks] == [("ru", "own"), ("uz", "own")]
    langs = speaker_languages(chunks, warnings)
    assert langs["S0"].language == "ru" and abs(langs["S0"].confidence - 5.3 / 10.4) < 0.01
    assert [x["language"] for x in langs["S0"].languages] == ["ru", "uz"]
    assert "lang_switch_detected" in warnings


def test_short_low_confidence_chunk_inherits_nearest_neighbour():
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(0, 1, 5.2, 6.5), _ch(0, 2, 7.0, 12.0)]
    lid = FakeLid({(0.0, 5.0): {"uz": 0.0, "ru": 0.999, "en": 0.001}, (5.2, 6.5): {"uz": 0.1, "ru": 0.13, "en": 0.77},
                   (7.0, 12.0): {"uz": 0.0, "ru": 0.998, "en": 0.002}})
    label_chunks(lid, AUDIO, chunks, [])
    # ближайший сосед — предыдущий (0.2 с против 0.5 с); уверенность — что аудио куска говорит про ru
    assert chunks[1].language == "ru" and chunks[1].lang_source == "prev" and chunks[1].lang_confidence == 0.13
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(0, 1, 5.5, 6.8), _ch(0, 2, 7.0, 12.0)]
    label_chunks(FakeLid({(0.0, 5.0): {"uz": 0.0, "ru": 0.999, "en": 0.001}, (5.5, 6.8): {"uz": 0.1, "ru": 0.13, "en": 0.77},
                          (7.0, 12.0): {"uz": 0.0, "ru": 0.998, "en": 0.002}}), AUDIO, chunks, [])
    assert chunks[1].language == "ru" and chunks[1].lang_source == "next"


def test_disagreeing_neighbours_resolved_by_own_audio():
    """Смена языка: ru-кусок, потом неуверенный (en 0.754 < 0.9) кусок, потом уверенный en.
    Соседи спорят — решает аудио самого куска (en), а не «сначала предыдущий»."""
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(0, 1, 5.2, 7.5), _ch(0, 2, 8.0, 10.6)]
    lid = FakeLid({(0.0, 5.0): {"uz": 0.0, "ru": 0.999, "en": 0.001}, (5.2, 7.5): {"uz": 0.1, "ru": 0.146, "en": 0.754},
                   (8.0, 10.6): {"uz": 0.02, "ru": 0.03, "en": 0.95}})
    label_chunks(lid, AUDIO, chunks, [])
    assert chunks[1].language == "en" and chunks[1].lang_source == "next" and chunks[1].lang_confidence == 0.754
    assert chunks[2].language == "en" and chunks[2].lang_source == "own"


def test_medium_chunk_below_threshold_inherits_when_neighbours_agree():
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(0, 1, 5.5, 7.8), _ch(0, 2, 8.0, 10.6)]
    lid = FakeLid({(0.0, 5.0): {"uz": 0.0, "ru": 0.999, "en": 0.001}, (5.5, 7.8): {"uz": 0.1, "ru": 0.146, "en": 0.754},
                   (8.0, 10.6): {"uz": 0.02, "ru": 0.95, "en": 0.03}})
    warnings = []
    label_chunks(lid, AUDIO, chunks, warnings)
    assert chunks[1].language == "ru" and chunks[1].lang_source == "next"
    # заметный кусок (2.3 с) получил язык против своего аудио (ru 0.146) — предупреждение
    assert "lang_ambiguous_chunks" in warnings


def test_adjacent_padded_chunks_overlapping_are_still_neighbours():
    """Соседние сегменты одного спикера режутся с паддингом и перекрываются на 0.3 с:
    ближайший решённый кусок — именно сосед, а не старый кусок за 30 с."""
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(1, 0, 4.85, 25.15), _ch(2, 0, 24.85, 40.0)]
    lid = FakeLid({(0.0, 5.0): {"uz": 0.0, "ru": 0.999, "en": 0.001}, (4.85, 25.15): {"uz": 0.99, "ru": 0.01, "en": 0.0},
                   (24.85, 40.0): {"uz": 0.85, "ru": 0.15, "en": 0.0}})
    label_chunks(lid, AUDIO, chunks, [])
    assert chunks[2].language == "uz" and chunks[2].lang_source == "prev"


def test_overlap_chunks_do_not_feed_speaker_prior():
    """Кусок с наложением слышит чужую речь: он не должен тянуть приор спикера."""
    chunks = [_ch(0, 0, 0.0, 3.0, "B"), _ch(1, 0, 40.0, 50.0, "B")]
    chunks[1].overlap = True
    lid = FakeLid({(0.0, 3.0): {"uz": 0.3, "ru": 0.7, "en": 0.0}, (40.0, 50.0): {"uz": 0.95, "ru": 0.05, "en": 0.0}})
    label_chunks(lid, AUDIO, chunks, [])
    assert chunks[0].language == "ru" and chunks[0].lang_source == "speaker_prior"


def test_short_chunk_keeps_own_label_when_very_confident():
    """Короткий кусок против единственного соседа: 0.985 недостаточно (наследует), 0.996 — решает сам."""
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(0, 1, 5.5, 7.0)]
    lid = FakeLid({(0.0, 5.0): {"uz": 0.0, "ru": 0.999, "en": 0.001}, (5.5, 7.0): {"uz": 0.985, "ru": 0.01, "en": 0.005}})
    label_chunks(lid, AUDIO, chunks, [])
    assert chunks[1].language == "ru" and chunks[1].lang_source == "prev"
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(0, 1, 5.5, 7.0)]
    lid = FakeLid({(0.0, 5.0): {"uz": 0.0, "ru": 0.999, "en": 0.001}, (5.5, 7.0): {"uz": 0.996, "ru": 0.003, "en": 0.001}})
    label_chunks(lid, AUDIO, chunks, [])
    assert chunks[1].language == "uz" and chunks[1].lang_source == "own"


def test_next_neighbour_when_no_previous():
    chunks = [_ch(0, 0, 0.0, 1.5), _ch(0, 1, 2.0, 7.0)]
    lid = FakeLid({(0.0, 1.5): {"uz": 0.5, "ru": 0.3, "en": 0.2}, (2.0, 7.0): {"uz": 0.0, "ru": 0.99, "en": 0.01}})
    label_chunks(lid, AUDIO, chunks, [])
    assert chunks[0].language == "ru" and chunks[0].lang_source == "next"


def test_far_neighbour_falls_back_to_soft_speaker_prior():
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(1, 0, 45.0, 46.5)]
    lid = FakeLid({(0.0, 5.0): {"uz": 0.05, "ru": 0.95, "en": 0.0}, (45.0, 46.5): {"uz": 0.6, "ru": 0.4, "en": 0.0}})
    label_chunks(lid, AUDIO, chunks, [])
    # донор дальше 30 с → мягкий приор: (0.05*5+0.6*1.5)/(6.5)=0.18 uz против ru (0.95*5+0.4*1.5)/6.5=0.82
    assert chunks[1].language == "ru" and chunks[1].lang_source == "speaker_prior"


def test_overlap_chunk_never_decides_on_its_own():
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(1, 0, 5.5, 9.0)]
    chunks[1].overlap = True
    lid = FakeLid({(0.0, 5.0): {"uz": 0.0, "ru": 0.999, "en": 0.001}, (5.5, 9.0): {"uz": 0.99, "ru": 0.01, "en": 0.0}})
    label_chunks(lid, AUDIO, chunks, [])
    assert chunks[1].language == "ru" and chunks[1].lang_source == "prev"


def test_speaker_too_short_uses_file_majority_then_default():
    chunks = [_ch(0, 0, 0.0, 6.0, "A"), _ch(1, 0, 10.0, 10.9, "B")]
    lid = FakeLid({(0.0, 6.0): {"uz": 0.99, "ru": 0.01, "en": 0.0}})
    warnings = []
    label_chunks(lid, AUDIO, chunks, warnings)
    assert chunks[1].language == "uz" and chunks[1].lang_source == "file_majority"
    lid2 = FakeLid({})
    only_short = [_ch(0, 0, 0.0, 0.9, "A")]
    label_chunks(lid2, AUDIO, only_short, warnings)
    assert only_short[0].language == "uz" and only_short[0].lang_source == "default" and "speaker_too_short_for_lid" in warnings


def test_lid_uses_chunk_audio_slices():
    chunks = [_ch(0, 0, 2.0, 6.0)]
    lid = FakeLid({(2.0, 6.0): {"uz": 1.0, "ru": 0.0, "en": 0.0}})
    label_chunks(lid, AUDIO, chunks, [])
    assert lid.calls == [4.0]


def test_speaker_languages_count_speech_inside_segment_only():
    a = Chunk(seg_id=0, idx=0, start=0.85, end=5.15, speaker="S0", seg_start=1.0, seg_end=5.0, language="ru")
    b = Chunk(seg_id=1, idx=0, start=9.85, end=12.15, speaker="S0", seg_start=10.0, seg_end=12.0, language="uz")
    langs = speaker_languages([a, b], [])
    assert {x["language"]: x["speech_sec"] for x in langs["S0"].languages} == {"ru": 4.0, "uz": 2.0}
    assert langs["S0"].confidence == round(4.0 / 6.0, 3)


def test_progress_callback_reports_chunk_counts():
    chunks = [_ch(0, 0, 0.0, 2.0), _ch(0, 1, 3.0, 6.0)]
    lid = FakeLid({(0.0, 2.0): {"uz": 1.0, "ru": 0.0, "en": 0.0}, (3.0, 6.0): {"uz": 1.0, "ru": 0.0, "en": 0.0}})
    seen = []
    label_chunks(lid, AUDIO, chunks, [], on_progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (2, 2)


def test_short_confident_chunk_against_all_neighbours_needs_very_high_confidence():
    """БЕНА 146.6–147.7: русский кусок 1.1 с, LID сказал en 0.985 между двумя ru — наследует ru;
    при 1.0 (mixed_06: английская вставка между uz) остаётся сам."""
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(0, 1, 5.3, 6.4), _ch(0, 2, 6.6, 12.0)]
    lid = FakeLid({(0.0, 5.0): {"uz": 0.0, "ru": 0.999, "en": 0.001}, (5.3, 6.4): {"uz": 0.005, "ru": 0.01, "en": 0.985},
                   (6.6, 12.0): {"uz": 0.0, "ru": 0.998, "en": 0.002}})
    label_chunks(lid, AUDIO, chunks, [])
    assert chunks[1].language == "ru" and chunks[1].lang_source == "next"
    chunks = [_ch(0, 0, 0.0, 5.0), _ch(0, 1, 5.3, 6.4), _ch(0, 2, 6.6, 12.0)]
    lid = FakeLid({(0.0, 5.0): {"uz": 0.99, "ru": 0.01, "en": 0.0}, (5.3, 6.4): {"uz": 0.0, "ru": 0.0, "en": 1.0},
                   (6.6, 12.0): {"uz": 0.99, "ru": 0.01, "en": 0.0}})
    label_chunks(lid, AUDIO, chunks, [])
    assert chunks[1].language == "en" and chunks[1].lang_source == "own"
