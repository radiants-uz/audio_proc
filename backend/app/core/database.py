from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# pool_pre_ping: соединения из пула проверяются перед выдачей, иначе после
# пересоздания контейнера Postgres api отвечает 500 «connection is closed»
# до собственного перезапуска.
async_engine = create_async_engine(
    settings.db.db_url, echo=settings.db.ECHO, pool_pre_ping=True, pool_recycle=1800
)

async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore
