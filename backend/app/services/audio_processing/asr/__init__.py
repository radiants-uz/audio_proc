from app.core.config import settings

from .base import AsrClient, AsrResult, AsrWord


def build_asr_client() -> AsrClient:
    if settings.asr.backend == "mock":
        from .mock import MockAsrClient

        return MockAsrClient()
    from .remote_api import RemoteAsrClient

    return RemoteAsrClient()


__all__ = ["AsrClient", "AsrResult", "AsrWord", "build_asr_client"]
