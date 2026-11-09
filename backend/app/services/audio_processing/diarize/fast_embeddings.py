"""Быстрые эмбеддинги для pyannote SpeakerDiarization: ResNet один раз на окно.

Стоковый ``SpeakerDiarization.get_embeddings`` для каждого 10-секундного окна
прогоняет ResNet (WeSpeaker) отдельно для каждого из 3 локальных спикеров, хотя
маска спикера участвует только в статистическом пулинге после свёрток. Здесь
свёртки (``forward_frames``) считаются один раз на окно, а пулинг с масками
(``forward_embedding`` с весами ``(batch, speakers, frames)``) — для всех
спикеров сразу. Окна, где все маски пустые, ResNet не проходят вовсе: при нулевых
весах пулинг даёт нули (``_pool`` добавляет 1e-8 к знаменателю), и эмбеддинг —
константа, не зависящая от аудио.

Результат совпадает со стоком, а прогонов ResNet втрое меньше. На CPU без AVX
эмбеддинги — почти всё время диаризации.
"""

import logging
import math
import types

import numpy as np

logger = logging.getLogger(__name__)


def install(pipeline) -> bool:
    """Подменяет ``get_embeddings`` у экземпляра пайплайна. False — модель не подходит."""
    model = getattr(getattr(pipeline, "_embedding", None), "model_", None)
    if not (hasattr(model, "forward_frames") and hasattr(model, "forward_embedding")):
        logger.warning(
            "fast embeddings not installed: %s has no forward_frames/forward_embedding",
            type(model).__name__,
        )
        return False
    stock = pipeline.get_embeddings

    def get_embeddings(self, file, binary_segmentations, exclude_overlap: bool = False, hook=None):
        # Кэш эмбеддингов при подборе гиперпараметров есть только в стоке.
        if self.training:
            return stock(file, binary_segmentations, exclude_overlap=exclude_overlap, hook=hook)
        return fast_embeddings(self, file, binary_segmentations, exclude_overlap, hook)

    pipeline.get_embeddings = types.MethodType(get_embeddings, pipeline)
    return True


def used_masks(binary_segmentations, exclude_overlap: bool, embedding) -> np.ndarray:
    """(chunks, speakers, frames): для каждой пары та же маска, что выбирает сток."""
    data = binary_segmentations.data
    _, num_frames, _ = data.shape
    masks = np.nan_to_num(data, nan=0.0).astype(np.float32)
    if exclude_overlap:
        num_samples = binary_segmentations.sliding_window.duration * embedding.sample_rate
        min_num_frames = math.ceil(num_frames * embedding.min_num_samples / num_samples)
        clean_frames = 1.0 * (np.sum(data, axis=2, keepdims=True) < 2)
        clean = np.nan_to_num(data * clean_frames, nan=0.0).astype(np.float32)
        masks = np.where(clean.sum(axis=1, keepdims=True) > min_num_frames, clean, masks)
    return np.ascontiguousarray(masks.transpose(0, 2, 1))


def fast_embeddings(
    self, file, binary_segmentations, exclude_overlap: bool = False, hook=None
) -> np.ndarray:
    import torch

    embedding = self._embedding
    model, device = embedding.model_, embedding.device
    window = binary_segmentations.sliding_window
    num_chunks, _, num_speakers = binary_segmentations.data.shape
    masks = used_masks(binary_segmentations, exclude_overlap, embedding)
    active = np.flatnonzero(masks.any(axis=(1, 2)))
    if len(active) == 0:  # pipeline выходит раньше, но на всякий случай
        active = np.arange(min(1, num_chunks))

    out = np.empty((num_chunks, num_speakers, embedding.dimension), dtype=np.float32)
    batch_size = max(1, int(self.embedding_batch_size))
    batch_count = math.ceil(len(active) / batch_size)
    if hook is not None:
        hook("embeddings", None, total=batch_count, completed=0)

    with torch.inference_mode():
        for b in range(batch_count):
            idx = active[b * batch_size : (b + 1) * batch_size]
            waveforms = torch.vstack(
                [self._audio.crop(file, window[i], mode="pad")[0][None] for i in idx]
            )
            frames = model.forward_frames(waveforms.to(device))
            weights = torch.from_numpy(masks[idx]).to(device)
            batch = model.forward_embedding(frames, weights=weights).cpu().numpy()
            if b == 0 and len(active) < num_chunks:
                zero = torch.zeros_like(weights[:1])
                out[:] = model.forward_embedding(frames[:1], weights=zero).cpu().numpy()
            out[idx] = batch
            if hook is not None:
                hook(
                    "embeddings",
                    batch.reshape(-1, batch.shape[-1]),
                    total=batch_count,
                    completed=b + 1,
                )

    logger.info(
        "embeddings: %d ResNet passes instead of %d", len(active), num_chunks * num_speakers
    )
    return out
