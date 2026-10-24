"""Automated integration tests for route prefix parity and endpoint aliases (Chunk A).

Verifies:
1. Prefix parity across /v1, /api/v1, and /api mounts for:
   - Health probes (GET /health, /v1/health, /api/v1/health, /api/health)
   - Daily rollups (GET /v1/days/{date}, /api/v1/days/{date}, /api/days/{date})
   - User profile (GET /v1/me/profile, /api/v1/me/profile, /api/me/profile)
   - Authentication (POST /v1/auth/login, /api/v1/auth/login)
2. Meal upload route aliases:
   - POST /v1/meals
   - POST /v1/meals/upload
   - POST /v1/meals/photo
   - POST /api/v1/meals
   - POST /api/v1/meals/upload
   - POST /api/v1/meals/photo
   - POST /api/meals/upload
   All invoke upload_meal handler returning HTTP 202 Accepted.
3. Legacy daily rollup route alias:
   - GET /v1/meals/daily-rollup?date=YYYY-MM-DD
   - GET /api/v1/meals/daily-rollup?date=YYYY-MM-DD
"""

from datetime import datetime, timezone
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.database import SessionLocal, sync_engine
from app.main import app
from app.models.base import Base
from app.models.triggers import register_i5_triggers
from scripts.ingest.seed_nutrition_data import seed_foods


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    register_i5_triggers(engine)

    SessionLocal.configure(bind=engine)

    with Session(engine) as session:
        seed_foods(session)
        yield session

    SessionLocal.configure(bind=sync_engine)
    Base.metadata.drop_all(engine)


@pytest.fixture
def client(test_db_session):
    """FastAPI TestClient with overridden get_db dependency."""
    def _override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client):
    """Register and authenticate an active user, returning authorization headers."""
    email = f"prefix_tester_{datetime.now(timezone.utc).timestamp()}@example.com"
    pwd = "SecurePassword123!"
    reg_res = client.post(
        "/v1/auth/register",
        json={"email": email, "password": pwd, "birth_year": 1998, "sex": "female"},
    )
    assert reg_res.status_code == 201
    token = reg_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_prefix_parity_health(client):
    """Verify health endpoints respond identically across all mounted prefixes."""
    endpoints = ["/health", "/v1/health", "/api/v1/health", "/api/health"]
    for ep in endpoints:
        res = client.get(ep)
        assert res.status_code == 200, f"Expected 200 from {ep}, got {res.status_code}"
        data = res.json()
        assert data.get("status") in ("healthy", "ok"), f"Unexpected payload from {ep}: {data}"


def test_route_aliases_meal_upload(client, auth_headers):
    """Verify POST /v1/meals, /upload, /photo and /api/v1 variants invoke upload_meal."""
    upload_paths = [
        "/v1/meals",
        "/v1/meals/upload",
        "/v1/meals/photo",
        "/api/v1/meals",
        "/api/v1/meals/upload",
        "/api/v1/meals/photo",
        "/api/meals/upload",
    ]

    for path in upload_paths:
        img = Image.new("RGB", (100, 100), color=(180, 50, 50))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = client.post(
            path,
            files={"file": ("test_alias.jpg", img_bytes, "image/jpeg")},
            data={"meal_slot": "lunch", "context": "personal_home"},
            headers=auth_headers,
        )
        assert response.status_code == 202, f"Expected 202 from {path}, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["status"] == "pending", f"Expected pending from {path}, got {data}"
        assert "meal_id" in data
        assert data["message"] == "Meal received and queued for analysis."


def test_prefix_parity_days_rollup(client, auth_headers):
    """Verify daily rollup endpoints respond identically across /v1 and /api/v1 prefixes."""
    today = datetime.now(timezone.utc).date().isoformat()

    paths = [
        f"/v1/days/{today}",
        f"/api/v1/days/{today}",
        f"/api/days/{today}",
        "/v1/days/today",
        "/api/v1/days/today",
    ]

    for p in paths:
        res = client.get(p, headers=auth_headers)
        assert res.status_code == 200, f"Expected 200 from {p}, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["date"] == today
        assert "total_kcal_mean" in data
        assert "meals" in data


def test_legacy_daily_rollup_alias(client, auth_headers):
    """Verify legacy /meals/daily-rollup?date=... query alias succeeds without 404."""
    today = datetime.now(timezone.utc).date().isoformat()

    legacy_paths = [
        f"/v1/meals/daily-rollup?date={today}",
        f"/api/v1/meals/daily-rollup?date={today}",
        "/v1/meals/daily-rollup",
        "/api/v1/meals/daily-rollup",
    ]

    for p in legacy_paths:
        res = client.get(p, headers=auth_headers)
        assert res.status_code == 200, f"Expected 200 from {p}, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["date"] == today
        assert "total_kcal_mean" in data


def test_prefix_parity_profile_and_auth(client, auth_headers):
    """Verify profile and auth endpoints work across /v1, /api/v1, and /api."""
    # Profile parity
    for p in ["/v1/me/profile", "/api/v1/me/profile", "/api/me/profile"]:
        res = client.get(p, headers=auth_headers)
        assert res.status_code == 200, f"Expected 200 from {p}, got {res.status_code}: {res.text}"
        data = res.json()
        assert "birth_year" in data
        assert "bmr" in data

    # Login parity
    email = "login_parity_tester@example.com"
    pwd = "ValidPassword123!"
    reg_res = client.post(
        "/v1/auth/register",
        json={"email": email, "password": pwd, "birth_year": 1992, "sex": "male"},
    )
    assert reg_res.status_code == 201

    for login_path in ["/v1/auth/login", "/api/v1/auth/login", "/api/auth/login"]:
        login_res = client.post(login_path, json={"email": email, "password": pwd})
        assert login_res.status_code == 200, f"Expected 200 from {login_path}, got {login_res.status_code}"
        assert "access_token" in login_res.json()
