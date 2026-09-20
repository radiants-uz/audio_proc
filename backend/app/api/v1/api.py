from fastapi import APIRouter, Depends

from app.api.deps import verify_admin_api_key

from .endpoints import audio, catalog, health

# Ключ требуется на весь /v1: здесь и загрузка файлов, и запуск платных
# LLM-задач. /health остаётся открытым — он объявлен на корне приложения.
router = APIRouter(dependencies=[Depends(verify_admin_api_key)])

router.include_router(audio.router, prefix="/audio", tags=["audio"])
router.include_router(catalog.services_router, prefix="/services", tags=["service"])
router.include_router(catalog.languages_router, prefix="/languages", tags=["language"])
router.include_router(catalog.operations_router, prefix="/operations", tags=["operation"])
router.include_router(health.router, prefix="/health", tags=["health"])
