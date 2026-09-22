from app.core.config import settings
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
