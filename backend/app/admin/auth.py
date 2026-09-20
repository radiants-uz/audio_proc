import secrets

from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request

from app.core.config import settings


class AdminAuth(AuthenticationBackend):
    """Простая аутентификация по логину/паролю из настроек."""

    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")

        if not isinstance(username, str) or not isinstance(password, str):
            return False

        expected_password = settings.admin.PASSWORD.get_secret_value()
        # Fail closed: пустой BACKEND_ADMIN_PASSWORD не должен пускать никого.
        if not expected_password:
            return False

        is_valid = username == settings.admin.USER and secrets.compare_digest(
            password.encode("utf-8"), expected_password.encode("utf-8")
        )
        if is_valid:
            request.session.update({"admin_auth": "true"})
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return request.session.get("admin_auth") == "true"
