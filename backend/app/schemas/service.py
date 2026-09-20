
from pydantic import BaseModel, ConfigDict


class ServiceRequest(BaseModel):
    operation: str | None = None
    language: str | None = None


class ServiceItem(BaseModel):
    name: str
    model_config = ConfigDict(from_attributes=True)


ServicesResponse = list[ServiceItem]


class LanguageRequest(BaseModel):
    service: str | None = None


class LanguageItem(BaseModel):
    language_code: str


LanguagesResponse = list[LanguageItem]


class OperationItem(BaseModel):
    operation_name: str


OperationsResponse = list[OperationItem]
