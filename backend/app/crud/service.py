
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.service import Service


async def get_service(db: AsyncSession, name: str) -> Service | None:
    result = await db.execute(select(Service).where(Service.name == name))
    return result.scalar_one_or_none()


async def list_services(
    db: AsyncSession, operation: str | None = None, language: str | None = None
) -> list[Service]:
    result = await db.execute(select(Service).where(Service.is_active.is_(True)).order_by(Service.name))
    services = list(result.scalars().all())
    if operation:
        # transcribe не использует LLM — сервисов у него нет.
        services = [s for s in services if operation in (s.operations or [])]
    if language and language != "auto":
        services = [s for s in services if not s.languages or language in s.languages]
    return services


async def list_languages(db: AsyncSession, service: str | None = None) -> list[str]:
    from app.core.fixtures import LanguageCode

    all_langs = [LanguageCode.UZ.value, LanguageCode.RU.value, LanguageCode.EN.value]
    if not service:
        return all_langs
    svc = await get_service(db, service)
    if not svc:
        return []
    return [lang for lang in all_langs if not svc.languages or lang in svc.languages]
