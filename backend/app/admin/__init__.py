from fastapi import FastAPI
from sqladmin import Admin

from app.core.config import settings
from app.core.database import async_session

from .auth import AdminAuth
from .views import VIEWS


def init_admin(app: FastAPI) -> None:
    authentication_backend = AdminAuth(secret_key=settings.admin.SECRET_KEY.get_secret_value())
    admin = Admin(app, session_maker=async_session, authentication_backend=authentication_backend)
    for view in VIEWS:
        admin.add_view(view)
