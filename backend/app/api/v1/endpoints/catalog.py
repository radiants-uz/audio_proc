from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import async_get_db
from app.core.fixtures import OperationEnum
from app.crud.service import list_languages, list_services
from app.schemas.service import (
    LanguageItem,
    LanguageRequest,
    LanguagesResponse,
    OperationItem,
    OperationsResponse,
    ServiceItem,
    ServiceRequest,
    ServicesResponse,
)

services_router = APIRouter()
languages_router = APIRouter()
operations_router = APIRouter()


@services_router.get("", response_model=ServicesResponse)
async def read_services(
    service_request: ServiceRequest = Depends(), db: AsyncSession = Depends(async_get_db)
):
    services = await list_services(db, service_request.operation, service_request.language)
    return [ServiceItem(name=s.name) for s in services]


@languages_router.get("", response_model=LanguagesResponse)
async def read_languages(
    language_request: LanguageRequest = Depends(), db: AsyncSession = Depends(async_get_db)
):
    return [LanguageItem(language_code=code) for code in await list_languages(db, language_request.service)]


@operations_router.get("", response_model=OperationsResponse)
async def read_operations():
    return [OperationItem(operation_name=op.value) for op in OperationEnum]
