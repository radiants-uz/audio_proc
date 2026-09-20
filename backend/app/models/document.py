from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from app.models.base import Base


class Document(Base):
    """Файл в MinIO: исходное аудио или один из результатов
    (transcript.json / analysis.json / report.pdf|docx). Результаты
    ссылаются на исходник через parent_id."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    object_key = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    parent_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    parent = relationship("Document", remote_side=[id], back_populates="children")
    children = relationship("Document", back_populates="parent", cascade="save-update, merge")

    # transcript | analysis | report | source
    kind = Column(String(16), nullable=False, default="source")
    language_code = Column(String(8), nullable=True)
    service_name = Column(String(32), nullable=True)
    operation_name = Column(String(32), nullable=True)

    # Токены LLM, тайминги стадий, warnings — всё, что полезно для отладки/стоимости.
    usage = Column(JSON, nullable=True)

    def __str__(self):
        return self.name
