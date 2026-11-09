"""fast_embeddings даёт те же эмбеддинги, что стоковый SpeakerDiarization.get_embeddings.

Вместо ResNet — крошечная модель с тем же устройством: свёртки → кадры,
StatsPool pyannote с весами спикеров → линейная голова.
"""

import types

import numpy as np
import pytest
import torch
from pyannote.audio.core.io import Audio
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.core import SlidingWindow, SlidingWindowFeature

from app.services.audio_processing.diarize import fast_embeddings

SR = 16000
DURATION, STEP = 2.0, 0.5
NUM_FRAMES, NUM_SPEAKERS, DIM = 50, 3, 4


class TinyEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv1d(1, 3, kernel_size=400, stride=160)
        self.pool = StatsPool()
        self.head = torch.nn.Linear(6, DIM)
        self.passes = 0

    def forward_frames(self, waveforms):
        self.passes += waveforms.shape[0]
        return torch.tanh(self.conv(waveforms))

    def forward_embedding(self, frames, weights=None):
        return self.head(self.pool(frames, weights=weights))

    def forward(self, waveforms, weights=None):
        return self.forward_embedding(self.forward_frames(waveforms), weights=weights)


class FakeEmbedding:
    sample_rate = SR
    dimension = DIM
    min_num_samples = 640
    device = torch.device("cpu")

    def __init__(self):
        self.model_ = TinyEmbeddingModel().eval()

    def __call__(self, waveforms, masks=None):
        with torch.inference_mode():
            return self.model_(waveforms, weights=masks).numpy()


def make_pipeline(batch_size: int):
    pipe = types.SimpleNamespace(
        _embedding=FakeEmbedding(),
        _audio=Audio(sample_rate=SR, mono="downmix"),
        embedding_batch_size=batch_size,
        training=False,
    )
    pipe.get_embeddings = types.MethodType(SpeakerDiarization.get_embeddings, pipe)
    return pipe


def make_segmentations(num_chunks: int) -> SlidingWindowFeature:
    rng = np.random.default_rng(1)
    data = (rng.random((num_chunks, NUM_FRAMES, NUM_SPEAKERS)) > 0.6).astype(np.float32)
    data[2] = 0  # окно тишины: все маски пустые
    data[3, :, 1:] = 0  # окно с одним голосом
    data[4, :, 0] = 1  # перекрытие голосов на всём окне
    data[4, :, 1] = 1
    data[5, 10:20, 2] = np.nan  # NaN при частичной склейке
    return SlidingWindowFeature(data, SlidingWindow(start=0.0, duration=DURATION, step=STEP))


@pytest.mark.parametrize("exclude_overlap", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_same_embeddings_as_stock(exclude_overlap, batch_size):
    num_chunks = 9
    seconds = DURATION + STEP * (num_chunks - 1)
    file = {"waveform": torch.randn(1, int(seconds * SR)) * 0.1, "sample_rate": SR}
    segmentations = make_segmentations(num_chunks)

    pipe = make_pipeline(batch_size)
    stock = pipe.get_embeddings(file, segmentations, exclude_overlap=exclude_overlap)
    stock_passes = pipe._embedding.model_.passes

    pipe._embedding.model_.passes = 0
    assert fast_embeddings.install(pipe)
    fast = pipe.get_embeddings(file, segmentations, exclude_overlap=exclude_overlap)

    assert fast.shape == stock.shape == (num_chunks, NUM_SPEAKERS, DIM)
    np.testing.assert_allclose(fast, stock, rtol=0, atol=1e-6)
    assert stock_passes == num_chunks * NUM_SPEAKERS
    assert pipe._embedding.model_.passes == num_chunks - 1  # окно тишины пропущено


def test_not_installed_without_frame_split():
    pipe = types.SimpleNamespace(_embedding=types.SimpleNamespace(model_=torch.nn.Linear(1, 1)))
    assert not fast_embeddings.install(pipe)
    assert not hasattr(pipe, "get_embeddings")
