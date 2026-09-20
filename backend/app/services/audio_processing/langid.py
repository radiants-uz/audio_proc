"""§5.4: язык на спикера. speechbrain VoxLingua107, softmax по {uz, ru, en}."""

import logging
from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.services.audio_processing.ingest import SAMPLE_RATE
from app.services.audio_processing.types import Segment

logger = logging.getLogger(__name__)

# Метки VoxLingua107 — двухбуквенные ISO-коды ("uz: Uzbek" в label_encoder).
_LABEL_ALIASES = {"uz": ("uz", "uz: Uzbek"), "ru": ("ru", "ru: Russian"), "en": ("en", "en: English")}


@dataclass
class LangResult:
    language: str
    confidence: float


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
        return LangResult(language=self.allowed[best], confidence=float(probs[best]))


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
