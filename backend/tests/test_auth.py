import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import settings
from app.main import app

API_KEY = "test-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings.admin, "API_KEY", SecretStr(API_KEY))
    # Без контекстного менеджера: иначе TestClient поднимет lifespan, а тот
    # полезет в Redis и arq. Авторизация отрабатывает до всяких зависимостей.
    return TestClient(app, raise_server_exceptions=False)


PROTECTED = "/v1/audio/tasks/some-task-id"


class TestApiKeyAuth:
    def test_health_stays_public(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_request_without_key_is_rejected(self, client):
        resp = client.get(PROTECTED)
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "X-API-Key"
        assert resp.json() == {"detail": "Invalid or missing API key"}

    def test_request_with_wrong_key_is_rejected(self, client):
        resp = client.get(PROTECTED, headers={"X-API-Key": "nope"})
        assert resp.status_code == 401

    def test_operations_catalog(self, client):
        resp = client.get("/v1/operations", headers={"X-API-Key": API_KEY})
        assert resp.status_code == 200
        assert [o["operation_name"] for o in resp.json()] == ["transcribe", "analyze", "report"]

    def test_unconfigured_key_fails_closed(self, client, monkeypatch):
        monkeypatch.setattr(settings.admin, "API_KEY", SecretStr(""))
        resp = client.get(PROTECTED, headers={"X-API-Key": API_KEY})
        assert resp.status_code == 500
