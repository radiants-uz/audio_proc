from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

async_engine = create_async_engine(settings.db.db_url, echo=settings.db.ECHO)

async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore
