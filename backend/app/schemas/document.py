from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentBase(BaseModel):
    name: str = Field(..., max_length=255)
    object_key: str = Field(..., max_length=512)
    kind: str = Field("source", max_length=16)
    language_code: str | None = Field(None, max_length=8)
    service_name: str | None = Field(None, max_length=32)
    operation_name: str | None = Field(None, max_length=32)

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class DocumentResponse(DocumentBase):
    id: int
    parent_id: int | None = None
    created_at: datetime
    url: str | None = None
    usage: dict | None = None


DocumentListResponse = list[DocumentResponse]
