"""Адаптивное улучшение звука для ASR и LID (не для диаризации: эмбеддинги
спикеров считаются по исходному сигналу, шумодав их искажает).

Решение принимается по замерам ffmpeg:
- тихая запись (mean_volume ниже quiet_db)  → speechnorm (подтягивает речь, не шум);
- шумная запись (noise floor выше noisy_db) → afftdn (частотный шумодав), мягко;
- всегда: highpass 80 Гц (гул, микрофонные удары).
Чистые громкие записи проходят без изменений — на них шумодав только вредит
(проверено: «Десять» → «Осять» после afftdn на студийной записи)."""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

_MEAN_RE = re.compile(r"mean_volume:\s*(-?[\d.]+) dB")
_NOISE_RE = re.compile(r"Noise floor dB:\s*(-?[\d.]+)")


@dataclass
class AudioStats:
    mean_db: float
    noise_floor_db: float | None


def measure(wav: Path) -> AudioStats:
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-i", str(wav),
         "-af", "volumedetect,astats=measure_overall=Noise_floor:measure_perchannel=none",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=600,
    ).stderr
    mean = _MEAN_RE.search(out)
    noise = _NOISE_RE.search(out)
    return AudioStats(
        mean_db=float(mean.group(1)) if mean else -20.0,
        noise_floor_db=float(noise.group(1)) if noise and noise.group(1) != "-" else None,
    )


def build_filters(stats: AudioStats) -> list[str]:
    cfg = settings.enhance
    filters = ["highpass=f=80"]
    noisy = stats.noise_floor_db is not None and stats.noise_floor_db > cfg.noisy_db
    quiet = stats.mean_db < cfg.quiet_db
    if noisy:
        filters.append(f"afftdn=nr={cfg.denoise_nr}:nf={cfg.denoise_nf}:tn=1")
    if quiet:
        # speechnorm: expansion речи с лимитером; e — степень, r — скорость
        filters.append(f"speechnorm=e={cfg.speechnorm_expansion}:r=0.0005:l=1")
    return filters


def enhance_for_asr(wav16k: Path, out: Path) -> tuple[Path, list[str], AudioStats]:
    """Возвращает путь к wav для ASR (исходный, если обработка не нужна)."""
    stats = measure(wav16k)
    if not settings.enhance.enabled:
        return wav16k, [], stats
    filters = build_filters(stats)
    if filters == ["highpass=f=80"] and not settings.enhance.always_highpass:
        logger.info("enhance: clean recording (mean %.1f dB, noise %s) — skipped", stats.mean_db, stats.noise_floor_db)
        return wav16k, [], stats
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-v", "error", "-i", str(wav16k),
         "-af", ",".join(filters), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(out)],
        capture_output=True, text=True, timeout=1800, check=True,
    )
    logger.info("enhance: mean %.1f dB, noise %s → %s", stats.mean_db, stats.noise_floor_db, ",".join(filters))
    return out, filters, stats
