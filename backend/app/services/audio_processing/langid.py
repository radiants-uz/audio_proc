"""§5.4: язык речи. speechbrain VoxLingua107, softmax по {uz, ru, en}.

Два режима:
- по кускам (LANGID_CHUNK_LID=true, по умолчанию): язык решается для каждого
  куска ASR — один спикер может переключать языки. Куски с уверенной оценкой
  решают сами, неуверенные и короткие наследуют язык соседних кусков того же
  спикера, затем мягкий приор спикера (среднее вероятностей по времени), затем
  большинство файла. Замер 2026-09-24: 158/158 реальных кусков против 71 % у
  схемы «язык на спикера» (речь на двух языках одного человека давала uz=1.0).
- по спикеру (legacy, LANGID_CHUNK_LID=false): identify_speakers + recheck_segments."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.services.audio_processing.ingest import SAMPLE_RATE
from app.services.audio_processing.types import Chunk, Segment

logger = logging.getLogger(__name__)

# Метки VoxLingua107 — двухбуквенные ISO-коды ("uz: Uzbek" в label_encoder).
_LABEL_ALIASES = {"uz": ("uz", "uz: Uzbek"), "ru": ("ru", "ru: Russian"), "en": ("en", "en: English")}


@dataclass
class LangResult:
    language: str
    confidence: float
    probs: dict[str, float] | None = None
    # доли времени по языкам (только для спикеров в режиме по кускам)
    languages: list[dict] | None = None


class LanguageIdentifier:
    def __init__(self):
        self.model_name = settings.langid.model
        self.allowed = list(settings.langid.allowed)
        self._clf = None
        self._indices: dict[str, int] = {}

    def load(self) -> None:
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier

        device = settings.audio.device if settings.audio.device != "mps" else "cpu"
        self._clf = EncoderClassifier.from_hparams(source=self.model_name, run_opts={"device": device})
        self._torch = torch
        labels = self._clf.hparams.label_encoder.ind2lab
        lab2ind = {str(v).strip(): int(k) for k, v in labels.items()}
        for lang in self.allowed:
            for alias in _LABEL_ALIASES.get(lang, (lang,)):
                for key, ind in lab2ind.items():
                    if key == alias or key.startswith(alias + ":") or key.split(":")[0] == alias:
                        self._indices[lang] = ind
                        break
                if lang in self._indices:
                    break
            if lang not in self._indices:
                raise RuntimeError(f"language '{lang}' not found in {self.model_name} labels")
        logger.info("LID loaded: %s, indices=%s", self.model_name, self._indices)

    def classify(self, audio: np.ndarray) -> LangResult:
        assert self._clf is not None, "call load() first"
        wav = self._torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        with self._torch.no_grad():
            out = self._clf.classify_batch(wav)
        logits = out[0][0].detach().cpu().numpy().astype(np.float64)  # log-probs по 107 классам
        sub = np.array([logits[self._indices[lang]] for lang in self.allowed])
        sub = sub - sub.max()
        probs = np.exp(sub) / np.exp(sub).sum()
        best = int(np.argmax(probs))
        return LangResult(
            language=self.allowed[best], confidence=float(probs[best]),
            probs={lang: float(p) for lang, p in zip(self.allowed, probs, strict=True)},
        )


def _collect_audio(audio: np.ndarray, segs: list[Segment], max_sec: float) -> np.ndarray:
    chunks, total = [], 0.0
    for s in sorted(segs, key=lambda x: -x.duration):
        take = min(s.duration, max_sec - total)
        if take <= 0.2:
            break
        a, b = int(s.start * SAMPLE_RATE), int((s.start + take) * SAMPLE_RATE)
        chunks.append(audio[a:b])
        total += take
        if total >= max_sec:
            break
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def identify_speakers(
    lid: LanguageIdentifier, audio: np.ndarray, segments: list[Segment], warnings: list[str]
) -> dict[str, LangResult]:
    cfg = settings.langid
    speakers = sorted({s.speaker for s in segments})
    results: dict[str, LangResult] = {}
    low: list[str] = []

    for spk in speakers:
        own = [s for s in segments if s.speaker == spk]
        clean = [s for s in own if not s.overlap]
        sample = _collect_audio(audio, clean, cfg.audio_sec)
        if len(sample) / SAMPLE_RATE < cfg.min_audio_sec:
            sample = _collect_audio(audio, own, cfg.audio_sec)
        if len(sample) / SAMPLE_RATE < cfg.min_audio_sec:
            results[spk] = LangResult(language="", confidence=0.0)
            low.append(spk)
            continue
        res = lid.classify(sample)
        results[spk] = res
        if res.confidence < cfg.min_confidence:
            low.append(spk)

    # Язык большинства (по времени речи среди уверенных спикеров) — fallback.
    speech = {}
    for s in segments:
        r = results.get(s.speaker)
        if r and r.language and r.confidence >= cfg.min_confidence:
            speech[r.language] = speech.get(r.language, 0.0) + s.duration
    fallback = max(speech, key=speech.get) if speech else (cfg.allowed[0] if cfg.allowed else "ru")

    for spk in low:
        r = results[spk]
        if not r.language:
            results[spk] = LangResult(language=fallback, confidence=0.0)
            if "speaker_too_short_for_lid" not in warnings:
                warnings.append("speaker_too_short_for_lid")
        else:
            results[spk] = LangResult(language=fallback, confidence=r.confidence)
            if "lid_low_confidence" not in warnings:
                warnings.append("lid_low_confidence")
    return results


def recheck_segments(lid: LanguageIdentifier, audio: np.ndarray, segments: list[Segment]) -> int:
    """§5.4 п.6: код-свитчинг по сегментам длиннее recheck_min_segment_sec."""
    cfg = settings.langid
    changed = 0
    for seg in segments:
        if seg.duration < cfg.recheck_min_segment_sec or seg.overlap:
            continue
        a, b = int(seg.start * SAMPLE_RATE), int(seg.end * SAMPLE_RATE)
        res = lid.classify(audio[a:b])
        if res.language != seg.language and res.confidence >= cfg.recheck_min_confidence:
            seg.language = res.language
            changed += 1
    return changed


# ---------------------------------------------------------------------------
# Режим по кускам
# ---------------------------------------------------------------------------

def _decided(chunk: Chunk, cfg) -> bool:
    if chunk.overlap or chunk.probs is None:
        return False
    if chunk.duration >= cfg.chunk_min_sec:
        return chunk.lang_confidence >= cfg.chunk_min_confidence
    return chunk.lang_confidence >= cfg.chunk_short_confidence


def _nearest(ch: Chunk, decided: list[Chunk], max_gap: float) -> tuple[Chunk | None, Chunk | None]:
    """Ближайшие решённые куски того же спикера до и после ch (по зазору между
    краями; соседние куски одного спикера могут перекрываться на паддинг)."""
    prev = nxt = None
    prev_gap = nxt_gap = max_gap + 1.0
    for d in decided:
        if d is ch:
            continue
        if (d.start, d.end) < (ch.start, ch.end):
            gap = max(0.0, ch.start - d.end)
            if gap <= max_gap and gap <= prev_gap:
                prev, prev_gap = d, gap
        else:
            gap = max(0.0, d.start - ch.end)
            if gap <= max_gap and gap < nxt_gap:
                nxt, nxt_gap = d, gap
    return prev, nxt


def _inherit(ch: Chunk, donor: Chunk, source: str) -> None:
    # уверенность унаследованного куска — что говорит ЕГО аудио про этот язык
    # (а не уверенность донора): низкое значение = аудио спорит с наследованием.
    probs = ch.probs or {}
    conf = probs[donor.language] if donor.language in probs else donor.lang_confidence
    ch.language, ch.lang_confidence, ch.lang_source = donor.language, round(conf, 4), source


def label_chunks(
    lid: LanguageIdentifier, audio: np.ndarray, chunks: list[Chunk], warnings: list[str],
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Язык каждого куска: своя оценка → ближайший решённый кусок того же спикера
    (при разногласии соседей — по вероятностям самого куска) → приор спикера →
    большинство файла → allowed[0]. Пишет chunk.language / lang_confidence /
    lang_source / probs. Куски с наложением речи не решают сами и не входят в приоры."""
    cfg = settings.langid
    allowed = list(lid.allowed)

    # 1. оценка по аудио
    for i, ch in enumerate(chunks, 1):
        if ch.duration >= cfg.chunk_lid_min_sec:
            a, b = int(ch.start * SAMPLE_RATE), int(ch.end * SAMPLE_RATE)
            res = lid.classify(audio[a:b])
            ch.probs = res.probs
            ch.language, ch.lang_confidence = res.language, res.confidence
            ch.lang_source = "own" if _decided(ch, cfg) else ""
        if on_progress and (i % 20 == 0 or i == len(chunks)):
            on_progress(i, len(chunks))

    # 2. наследование от ближайшего решённого куска того же спикера
    by_speaker: dict[str, list[Chunk]] = {}
    for ch in chunks:
        by_speaker.setdefault(ch.speaker, []).append(ch)
    for items in by_speaker.values():
        items.sort(key=lambda c: (c.start, c.end))
        decided = [c for c in items if c.lang_source == "own"]
        # Короткий кусок, который спорит со всеми решёнными соседями, — чаще ошибка LID на
        # 1–2 с аудио, чем настоящая вставка на другом языке: без очень высокой уверенности
        # он теряет право решать сам и наследует, как остальные.
        demoted = []
        for ch in decided:
            if ch.duration >= cfg.chunk_min_sec or ch.lang_confidence >= cfg.chunk_short_against_confidence:
                continue
            prev, nxt = _nearest(ch, decided, cfg.neighbour_max_gap_sec)
            neighbours = [x for x in (prev, nxt) if x is not None]
            if neighbours and all(x.language != ch.language for x in neighbours):
                demoted.append(ch)
        for ch in demoted:
            ch.lang_source = ""
            decided.remove(ch)
        for ch in items:
            if ch.lang_source == "own":
                continue
            prev, nxt = _nearest(ch, decided, cfg.neighbour_max_gap_sec)
            if prev is None and nxt is None:
                continue
            if prev is None or nxt is None or prev.language == nxt.language:
                donor, source = (prev, "prev") if prev is not None else (nxt, "next")
                if prev is not None and nxt is not None:
                    donor, source = (prev, "prev") if ch.start - prev.end <= nxt.start - ch.end else (nxt, "next")
            else:
                # соседи спорят: решает аудио самого куска, при равенстве — ближний
                probs = ch.probs or {}
                p_prev, p_next = probs.get(prev.language, 0.0), probs.get(nxt.language, 0.0)
                if p_prev == p_next:
                    donor, source = (prev, "prev") if ch.start - prev.end <= nxt.start - ch.end else (nxt, "next")
                else:
                    donor, source = (prev, "prev") if p_prev > p_next else (nxt, "next")
            _inherit(ch, donor, source)

    # 3. мягкий приор спикера (среднее вероятностей по чистым кускам, взвешенное по
    #    длительности); куски с наложением — только если чистых нет
    file_mass: dict[str, float] = dict.fromkeys(allowed, 0.0)
    file_mass_overlap: dict[str, float] = dict.fromkeys(allowed, 0.0)
    for items in by_speaker.values():
        clean = [c for c in items if c.probs and not c.overlap]
        sample = clean or [c for c in items if c.probs]
        mass = dict.fromkeys(allowed, 0.0)
        classified = 0.0
        for c in sample:
            classified += c.duration
            for lang, p in c.probs.items():
                mass[lang] = mass.get(lang, 0.0) + p * c.duration
        target = file_mass if clean else file_mass_overlap
        for lang, m in mass.items():
            target[lang] += m
        undecided = [c for c in items if not c.lang_source]
        if not undecided:
            continue
        if classified >= cfg.prior_min_sec:
            lang = max(mass, key=mass.get)
            conf = mass[lang] / classified
            for c in undecided:
                c.language, c.lang_confidence, c.lang_source = lang, round(conf, 4), "speaker_prior"

    # 4. большинство файла, затем язык по умолчанию
    still = [c for c in chunks if not c.lang_source]
    if still:
        pool = file_mass if sum(file_mass.values()) > 0 else file_mass_overlap
        total = sum(pool.values())
        if total > 0:
            lang = max(pool, key=pool.get)
            conf = pool[lang] / total
            for c in still:
                c.language, c.lang_confidence, c.lang_source = lang, round(conf, 4), "file_majority"
        else:
            for c in still:
                c.language, c.lang_confidence, c.lang_source = allowed[0], 0.0, "default"
            if "speaker_too_short_for_lid" not in warnings:
                warnings.append("speaker_too_short_for_lid")

    # Заметный кусок (≥ chunk_min_sec) получил язык не по своему аудио, и аудио с ним
    # не согласно (вероятность этого языка < 0.5) — предупредить клиента.
    ambiguous = [
        c for c in chunks
        if c.lang_source not in ("own", "") and c.duration >= cfg.chunk_min_sec and c.lang_confidence < 0.5
    ]
    if ambiguous and "lang_ambiguous_chunks" not in warnings:
        warnings.append("lang_ambiguous_chunks")
    logger.info(
        "chunk LID: %d chunks, own=%d prev/next=%d prior=%d majority=%d default=%d ambiguous=%d",
        len(chunks), sum(c.lang_source == "own" for c in chunks),
        sum(c.lang_source in ("prev", "next") for c in chunks), sum(c.lang_source == "speaker_prior" for c in chunks),
        sum(c.lang_source == "file_majority" for c in chunks), sum(c.lang_source == "default" for c in chunks),
        len(ambiguous),
    )


def speaker_languages(chunks: list[Chunk], warnings: list[str] | None = None, min_switch_sec: float = 2.0) -> dict[str, LangResult]:
    """Доминирующий язык спикера = большинство по времени его кусков (время речи
    внутри сегментов, без паддинга); confidence = доля этого языка;
    languages — раскладка по языкам."""
    out: dict[str, LangResult] = {}
    by_speaker: dict[str, dict[str, float]] = {}
    for ch in chunks:
        if not ch.language:
            continue
        d = by_speaker.setdefault(ch.speaker, {})
        d[ch.language] = d.get(ch.language, 0.0) + ch.speech_sec
    for spk, d in by_speaker.items():
        total = sum(d.values()) or 1.0
        lang = max(d, key=d.get)
        breakdown = [
            {"language": k, "speech_sec": round(v, 2), "share": round(v / total, 3)}
            for k, v in sorted(d.items(), key=lambda kv: -kv[1])
        ]
        out[spk] = LangResult(language=lang, confidence=round(d[lang] / total, 3), languages=breakdown)
        if warnings is not None and sum(1 for v in d.values() if v >= min_switch_sec) >= 2:
            if "lang_switch_detected" not in warnings:
                warnings.append("lang_switch_detected")
    return out
