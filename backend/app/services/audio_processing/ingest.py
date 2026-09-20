"""§5.1: любой контейнер → 16 kHz mono PCM16 WAV через ffmpeg."""

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


class IngestError(Exception):
    pass


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
        return float(json.loads(out)["format"]["duration"])
    except Exception:  # noqa: BLE001
        return None


def to_wav16k(input_path: Path, output_path: Path) -> float:
    """Возвращает длительность (сек)."""
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-v", "error",
        "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", "-f", "wav",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=True)
    except subprocess.CalledProcessError as e:
        raise IngestError(f"ffmpeg failed: {(e.stderr or '').strip()[:500] or 'unknown error'}") from e
    except FileNotFoundError as e:
        raise IngestError("ffmpeg is not installed") from e
    info = sf.info(str(output_path))
    return float(info.frames) / float(info.samplerate)


def load_wav(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise IngestError(f"unexpected sample rate {sr}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def slice_to_wav_bytes(audio: np.ndarray, start: float, end: float) -> bytes:
    import io

    a = max(0, int(start * SAMPLE_RATE))
    b = min(len(audio), int(end * SAMPLE_RATE))
    buf = io.BytesIO()
    sf.write(buf, audio[a:b], SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()
