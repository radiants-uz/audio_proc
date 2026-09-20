from contextlib import asynccontextmanager

from arq import create_pool
from fastapi import FastAPI

from app.core.redis import get_arq_redis_settings, get_redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = get_redis_client(decode_responses=True)
    arq_pool = await create_pool(get_arq_redis_settings())

    app.state.redis = redis_client
    app.state.arq_pool = arq_pool

    yield

    await redis_client.aclose()
    await arq_pool.aclose()
