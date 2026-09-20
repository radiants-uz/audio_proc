import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.admin import init_admin
from app.api.v1.api import router as v1_router
from app.core.config import settings
from app.core.lifespan import lifespan
from app.exceptions.base import AppException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

app = FastAPI(
    title="Audio Processing API",
    version="1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allow_origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail or "Application error"},
    )


app.include_router(v1_router, prefix="/v1")

init_admin(app)
