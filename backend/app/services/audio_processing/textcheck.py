"""Проверка правдоподобия текста ASR: «этот текст действительно на языке,
которым его распознали?».

Зачем: русская речь через узбекскую модель возвращается НЕ пустой, а
фонетической транслитерацией («Zagruziti zapes Strechi Lelizvanka…»), и
пустой ответ как сигнал ошибки не срабатывает (0/70 таких случаев пусты).

Как: символьные 5-граммные модели (Kneser–Ney-подобное абсолютное
дисконтирование) для uz/ru/en в латинице и кириллице, обученные на текстах
FLEURS train (по 2000 предложений на язык; ru_lat/uz_cyr — транслитерации).
Правило (подобрано на 370 парах, 0 ошибок; см. STT_PIPELINE_PLAN.md §5.6):
  текст «не того языка», если пустой ИЛИ (при ≥ min_tokens слов)
  score(ожидаемая модель) < LM_ABS ИЛИ score(ожидаемая) − score(лучшая чужая)
  < LM_MARGIN[lang] ИЛИ доля чужого алфавита > FOREIGN_SCRIPT ИЛИ плотность
  текста < min_cps символов/с. На коротких ответах (< min_tokens) LM-правила
  ненадёжны («Кстати.» под ru имеет margin 0.48) — там только алфавит и плотность.
Таблицы: data/ngram_tables_n5_mc2.json.gz (1.1 МБ), без загрузок в рантайме.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_TABLES_PATH = Path(__file__).parent / "data" / "ngram_tables_n5_mc2.json.gz"

_APOS = "ʻʼ'’`‘´"
_apos_re = re.compile("[" + re.escape(_APOS) + "]")
_nonword_re = re.compile(r"[^\w\sʻ]", re.U)
_PUNCT = set(".,!?;:…-–—\"'()[]«» \t\n")

# Какая модель «своя» для кода языка (uz-модель пишет латиницей, ru — кириллицей).
EXPECTED = {"uz": "uz_lat", "ru": "ru_cyr", "en": "en_lat"}
# Подсказка, каким языком повторить, по лучшей чужой модели.
HINT = {"uz_lat": "uz", "uz_cyr": "uz", "ru_lat": "ru", "ru_cyr": "ru", "en_lat": "en", "en_cyr": "en"}

# Пороги (биты на символ), подобранные в исследовании 2026-09-24:
# зазор margin uz [1.146, 2.408], ru [1.785, 2.115]; худший верный абсолютный счёт −4.853.
LM_ABS = {"uz": -5.5, "ru": -5.5, "en": -5.5}
LM_MARGIN = {"uz": 1.8, "ru": 1.95, "en": 1.8}
FOREIGN_SCRIPT = 0.5
# Средний счёт СВОЕЙ модели на верных транскриптах (uz/ru — 67/70 реальных ответов
# ASR, en — 10 предложений): модели по-разному «щедры», и сравнивать сырые биты
# между моделями нельзя (ru_cyr на ~0.4 бита/символ ниже uz_lat). pick() вычитает это.
LM_BASELINE = {"uz_lat": -2.38, "ru_cyr": -2.76, "en_lat": -2.42}
# Кандидат заменяет исходный текст, только если правдоподобнее его на столько бит/символ
# (после поправки на baseline). Транслитерация проигрывает верному тексту на 3+ бита.
PICK_MIN_GAIN = 0.5
# Правило плотности (символов/с) — только на кусках не короче этого: односложный ответ
# («Да.», «Ha.») в куске 1–2 с нормален, а не галлюцинация.
DENSITY_MIN_SEC = 2.0


def has_words(text: str | None) -> bool:
    """ASR на шуме часто возвращает «.» — это не текст, а пустой ответ."""
    return bool(text) and any(ch not in _PUNCT for ch in text)


def normalize(text: str | None) -> str:
    t = unicodedata.normalize("NFC", text or "").lower()
    t = _apos_re.sub("ʻ", t).replace("ё", "е")
    t = _nonword_re.sub(" ", t)
    t = re.sub(r"\d+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class _CharLM:
    LAMBDA = 0.02

    def __init__(self, j: dict):
        self.n, self.d, self.vocab = j["n"], j["d"], j["vocab"]
        self.tables = [{ctx: (sum(c.values()), len(c), c) for ctx, c in tab.items()} for tab in j["tables"]]

    def _p(self, ctx: str, ch: str, k: int) -> float:
        if k == 0:
            tot, _, cnt = self.tables[0][""]
            return (1 - self.LAMBDA) * cnt.get(ch, 0) / tot + self.LAMBDA / self.vocab
        ent = self.tables[k].get(ctx[len(ctx) - k :])
        lower = self._p(ctx, ch, k - 1)
        if not ent:
            return lower
        tot, types, cnt = ent
        return max(cnt.get(ch, 0) - self.d, 0) / tot + (self.d * types / tot) * lower

    def score(self, s: str) -> float:
        """Средний log2 P(символ) — биты на символ (чем выше, тем правдоподобнее)."""
        s2 = "\x02" * (self.n - 1) + s + "\x03"
        lp = sum(math.log2(self._p(s2[i - self.n + 1 : i], s2[i], self.n - 1)) for i in range(self.n - 1, len(s2)))
        return lp / (len(s2) - self.n + 1)


@dataclass
class CheckResult:
    wrong: bool
    reasons: list[str] = field(default_factory=list)
    suggest: str | None = None
    lm_expected: float | None = None
    lm_margin: float | None = None
    foreign: float | None = None
    cps: float | None = None
    tokens: int = 0

    @property
    def reason(self) -> str:
        return ",".join(self.reasons)


class TextPlausibility:
    """Загружается один раз в воркере (ModelRegistry.textcheck)."""

    def __init__(self, tables_path: Path = _TABLES_PATH):
        self._path = tables_path
        self._lms: dict[str, _CharLM] = {}

    def load(self) -> None:
        with gzip.open(self._path, "rt", encoding="utf-8") as f:
            self._lms = {k: _CharLM(v) for k, v in json.load(f).items()}
        logger.info("text plausibility tables loaded: %s", sorted(self._lms))

    @property
    def loaded(self) -> bool:
        return bool(self._lms)

    def scores(self, text: str) -> dict[str, float]:
        s = normalize(text)
        return {k: lm.score(s) for k, lm in self._lms.items()} if s else {}

    def check(
        self, text: str, lang: str, audio_sec: float | None, min_tokens: int = 3, min_cps: float = 4.0,
        min_density_sec: float = DENSITY_MIN_SEC,
    ) -> CheckResult:
        if not has_words(text):
            return CheckResult(wrong=True, reasons=["empty"])
        if lang not in EXPECTED or not self.loaded:
            return CheckResult(wrong=False)
        norm = normalize(text)
        tokens = len(norm.split())
        sc = self.scores(text)
        if not sc:
            # только цифры/символы («2024.», «10%», «+7 900…»): язык по ним не определить — не трогаем
            return CheckResult(wrong=False, tokens=0)
        exp = EXPECTED[lang]
        other = max((k for k in sc if k != exp), key=sc.get)
        margin = sc[exp] - sc[other]
        letters = [c for c in norm if c.isalpha()]
        cyr = sum("Ѐ" <= c <= "ӿ" for c in letters) / max(1, len(letters))
        foreign = (1 - cyr) if exp.endswith("_cyr") else cyr
        cps = len(text) / audio_sec if audio_sec else None
        reasons: list[str] = []
        if tokens >= min_tokens:
            if sc[exp] < LM_ABS[lang]:
                reasons.append("lm_abs")
            if margin < LM_MARGIN[lang]:
                reasons.append("lm_margin")
        if foreign > FOREIGN_SCRIPT:
            reasons.append("foreign_script")
        if cps is not None and audio_sec >= min_density_sec and cps < min_cps:
            reasons.append("low_density")
        return CheckResult(
            wrong=bool(reasons), reasons=reasons, suggest=HINT[other] if reasons else None,
            lm_expected=round(sc[exp], 3), lm_margin=round(margin, 3), foreign=round(foreign, 2),
            cps=round(cps, 2) if cps is not None else None, tokens=tokens,
        )

    def pick(self, candidates: dict[str, str], original: str | None = None, min_gain: float = PICK_MIN_GAIN) -> str | None:
        """Из {язык: текст} выбирает язык, чей текст правдоподобнее для СВОЕЙ модели
        (счёт с поправкой на baseline модели). Если задан original (язык исходного
        текста среди кандидатов), другой язык побеждает только с запасом min_gain."""
        scored: dict[str, float] = {}
        for lang, text in candidates.items():
            if not has_words(text) or lang not in EXPECTED:
                continue
            model = EXPECTED[lang]
            s = self.scores(text).get(model)
            if s is not None:
                scored[lang] = s - LM_BASELINE.get(model, 0.0)
        if not scored:
            return None
        best = max(scored, key=scored.get)
        if original in scored and best != original and scored[best] - scored[original] < min_gain:
            return original
        return best
