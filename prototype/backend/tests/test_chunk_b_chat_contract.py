"""Automated integration tests for Chunk B: Chat Co-Pilot Contract & History Sync.

Verifies:
1. POST /v1/chat accepts either 'message' or 'prompt' or both.
2. Invalid/empty requests properly return HTTP 422.
3. GET /v1/chat/history returns persisted turns in chronological order.
4. Parity across /v1/chat and /api/v1/chat (and /api/chat).
5. Response payload conforms strictly to ChatResponse contract (reply and macros).
"""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
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
    """Create an isolated, in-memory SQLite database session."""
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
    email = f"chat_tester_{datetime.now(timezone.utc).timestamp()}@example.com"
    pwd = "SecurePassword123!"
    reg = client.post(
        "/v1/auth/register",
        json={"email": email, "password": pwd, "birth_year": 1995, "sex": "male"},
    )
    assert reg.status_code == 201
    token = reg.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_chat_with_message_key(client, auth_headers):
    """Verify POST /v1/chat succeeds when sending standard {'message': ...}."""
    resp = client.post(
        "/v1/chat",
        json={"message": "How much protein do I have left today?"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert "protein" in data["reply"].lower()
    assert "macros" in data
    macros = data["macros"]
    for k in ["target_kcal", "consumed_kcal", "remaining_kcal", "protein_g", "carb_g", "fat_g"]:
        assert k in macros
        assert isinstance(macros[k], (int, float))


def test_chat_with_prompt_key(client, auth_headers):
    """Verify POST /v1/chat succeeds when sending client {'prompt': ...} (Chunk B alias)."""
    resp = client.post(
        "/v1/chat",
        json={"prompt": "What should I eat for dinner?"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert "dinner" in data["reply"].lower()
    assert "macros" in data


def test_chat_with_both_message_and_prompt(client, auth_headers):
    """Verify POST /v1/chat succeeds when client sends both {'message': ..., 'prompt': ...}."""
    resp = client.post(
        "/v1/chat",
        json={"message": "How am I doing today?", "prompt": "How am I doing today?"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert "macros" in data


def test_chat_empty_or_invalid_payload_rejected(client, auth_headers):
    """Verify empty or missing text payload returns HTTP 422."""
    # Completely empty payload
    resp = client.post("/v1/chat", json={}, headers=auth_headers)
    assert resp.status_code == 422

    # Empty string message
    resp = client.post("/v1/chat", json={"message": ""}, headers=auth_headers)
    assert resp.status_code == 422

    # Empty string prompt
    resp = client.post("/v1/chat", json={"prompt": ""}, headers=auth_headers)
    assert resp.status_code == 422


def test_chat_history_order_and_persistence(client, auth_headers):
    """Verify GET /v1/chat/history returns persisted turns in chronological order."""
    # Send 2 questions
    q1 = "First question: How much protein left?"
    q2 = "Second question: What to eat for dinner?"

    r1 = client.post("/v1/chat", json={"prompt": q1}, headers=auth_headers)
    assert r1.status_code == 200

    r2 = client.post("/v1/chat", json={"message": q2}, headers=auth_headers)
    assert r2.status_code == 200

    # Retrieve history
    hist_resp = client.get("/v1/chat/history", headers=auth_headers)
    assert hist_resp.status_code == 200
    history = hist_resp.json()

    # We expect 4 turns: user1, bot1, user2, bot2
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[0]["message"] == q1
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "user"
    assert history[2]["message"] == q2
    assert history[3]["role"] == "assistant"

    # Verify chronological ordering
    dt0 = datetime.fromisoformat(history[0]["created_at"])
    dt2 = datetime.fromisoformat(history[2]["created_at"])
    assert dt0 <= dt2


def test_chat_route_prefix_parity(client, auth_headers):
    """Verify /v1/chat, /api/v1/chat, and /api/chat all operate identically."""
    prefixes = ["/v1/chat", "/api/v1/chat", "/api/chat"]

    for path in prefixes:
        resp = client.post(
            path,
            json={"prompt": f"Testing parity on {path}"},
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"Failed POST on {path}: {resp.status_code} {resp.text}"
        data = resp.json()
        assert "reply" in data
        assert "macros" in data

    # Verify history parity
    for path in ["/v1/chat/history", "/api/v1/chat/history", "/api/chat/history"]:
        hist = client.get(path, headers=auth_headers)
        assert hist.status_code == 200, f"Failed GET on {path}: {hist.status_code}"
        assert len(hist.json()) >= 6  # 3 pairs sent above
