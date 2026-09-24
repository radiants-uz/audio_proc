from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.config import settings
from app.services.audio_processing import enhance as enh
from app.services.audio_processing.enhance import AudioStats, build_filters


def test_clean_loud_recording_untouched():
    assert build_filters(AudioStats(mean_db=-20, noise_floor_db=-60)) == ["highpass=f=80"]


def test_quiet_gets_speechnorm():
    f = build_filters(AudioStats(mean_db=-35, noise_floor_db=-60))
    assert any(x.startswith("speechnorm") for x in f) and not any(x.startswith("afftdn") for x in f)


def test_noisy_gets_denoise():
    f = build_filters(AudioStats(mean_db=-20, noise_floor_db=-30))
    assert any(x.startswith("afftdn") for x in f)


def test_quiet_and_noisy_get_both_in_order():
    f = build_filters(AudioStats(mean_db=-40, noise_floor_db=-30))
    assert [x.split("=")[0] for x in f] == ["highpass", "afftdn", "speechnorm"]


def test_unknown_noise_floor_is_not_noisy(monkeypatch):
    monkeypatch.setattr(settings.enhance, "quiet_db", -28.0)
    assert build_filters(AudioStats(mean_db=-20, noise_floor_db=None)) == ["highpass=f=80"]


def _wav(path: Path, seconds: float = 1.0, amp: float = 0.01) -> Path:
    x = (amp * np.sin(2 * np.pi * 220 * np.linspace(0, seconds, int(16000 * seconds)))).astype("float32")
    sf.write(str(path), x, 16000, subtype="PCM_16")
    return path


class FakeDenoiser:
    name = "deepfilternet3"
    loaded = True

    def denoise(self, audio):
        return audio * 0.5


class BrokenDenoiser(FakeDenoiser):
    def denoise(self, audio):
        raise RuntimeError("boom")


def test_denoiser_not_loaded_falls_back_to_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.enhance, "neural", "deepfilter")
    src = _wav(tmp_path / "audio16k.wav", amp=0.01)  # тихо → speechnorm
    out, applied, stats = enh.enhance_for_asr(src, tmp_path / "audio16k_enh.wav", denoiser=None)
    assert "deepfilternet3" not in applied and any(a.startswith("speechnorm") for a in applied)
    assert out.exists()


def test_denoiser_applied_then_filters(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.enhance, "neural", "deepfilter")
    src = _wav(tmp_path / "audio16k.wav", amp=0.01)
    out, applied, stats = enh.enhance_for_asr(src, tmp_path / "audio16k_enh.wav", denoiser=FakeDenoiser())
    assert applied[0] == "deepfilternet3" and (tmp_path / "audio16k_df.wav").exists()
    assert out.exists() and out != src


def test_broken_denoiser_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.enhance, "neural", "deepfilter")
    src = _wav(tmp_path / "audio16k.wav", amp=0.01)
    out, applied, stats = enh.enhance_for_asr(src, tmp_path / "audio16k_enh.wav", denoiser=BrokenDenoiser())
    assert "deepfilternet3" not in applied and out.exists()



def test_chunked_denoise_reassembles_signal_with_crossfade():
    from app.services.audio_processing.denoise import chunked

    sr = 16000
    x = np.random.default_rng(0).standard_normal(sr * 7).astype("float32")
    calls = []

    def identity(piece):
        calls.append(len(piece) / sr)
        return piece

    y = chunked(identity, x, sr, chunk_sec=2.0, overlap_sec=0.5)
    assert y.shape == x.shape and np.allclose(y, x, atol=1e-6)  # окна с кроссфейдом собираются в исходник
    assert len(calls) > 1 and max(calls) <= 2.5 + 1e-6
    # короткий файл — один вызов
    calls.clear()
    assert np.allclose(chunked(identity, x[: sr * 2], sr, 2.0, 0.5), x[: sr * 2]) and calls == [2.0]
