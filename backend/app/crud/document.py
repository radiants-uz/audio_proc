
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document


async def get_document(db: AsyncSession, doc_id: int) -> Document | None:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    return result.scalar_one_or_none()


async def get_children(db: AsyncSession, parent_id: int, kind: str | None = None) -> list[Document]:
    stmt = select(Document).where(Document.parent_id == parent_id)
    if kind:
        stmt = stmt.where(Document.kind == kind)
    stmt = stmt.order_by(Document.created_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def create_document(
    db: AsyncSession,
    name: str,
    object_key: str,
    kind: str = "source",
    language_code: str | None = None,
    service_name: str | None = None,
    operation_name: str | None = None,
    parent_id: int | None = None,
    usage: dict | None = None,
) -> Document:
    try:
        doc = Document(
            name=name,
            object_key=object_key,
            kind=kind,
            language_code=language_code,
            service_name=service_name,
            operation_name=operation_name,
            parent_id=parent_id,
            usage=usage,
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        return doc
    except Exception:
        await db.rollback()
        raise
