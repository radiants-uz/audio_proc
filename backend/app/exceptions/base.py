from typing import Any

from fastapi import status


class AppException(Exception):
    status_code: int = status.HTTP_400_BAD_REQUEST
    detail: str = "Bad request"

    def __init__(self, detail: str | None = None, **kwargs: Any):
        self.detail = detail or self.detail
        self.extra = kwargs

    def __str__(self):
        return self.detail


class FileTooLargeError(AppException):
    status_code: int = status.HTTP_413_CONTENT_TOO_LARGE
    detail: str = "File too large"


class ClientAppError(AppException):
    status_code: int = status.HTTP_400_BAD_REQUEST


class ServerAppError(AppException):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR


class NotFoundError(AppException):
    status_code: int = status.HTTP_404_NOT_FOUND
    detail: str = "Not found"


class TaskIDNotFoundError(NotFoundError):
    detail: str = "Task not found"


class ServiceUnavailableError(AppException):
    status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE
    detail: str = "Service unavailable"
