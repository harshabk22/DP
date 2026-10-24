"""Unit and integration tests for Chunk 5: Chat Persistence & 19:00 Dinner Gap Engine.

Verifies:
1. POST /v1/chat persists user and assistant turns to ChatMessage in DB.
2. GET /v1/chat/history returns persisted conversations in chronological order.
3. 19:00 Dinner Gap Engine calculates protein/fiber deficit and recommends non-shaming swaps.
4. GET /v1/days/{day_str} returns dinner_advice and dinner_swaps.
5. Invariant I8 non-shaming enforcement: no toxic language or guilt framing even when target is exceeded.
"""

from datetime import date, datetime, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.safety import validate_non_shaming_text
from app.main import app
from app.models.base import Base
from app.models.chat import ChatMessage
from app.models.food import Food
from app.models.meal import Meal, MealItem
from app.models.triggers import register_i5_triggers
from app.models.user import Profile, User
from app.services.digest.engine import WeeklyDigestEngine
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

    with Session(engine) as session:
        seed_foods(session)
        yield session

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
def test_user_token(client):
    """Register and authenticate an active user."""
    email = f"chat_tester_{datetime.now(timezone.utc).timestamp()}@example.com"
    pwd = "StrongPassword123!"
    reg = client.post(
        "/v1/auth/register",
        json={"email": email, "password": pwd, "birth_year": 1998, "sex": "female"},
    )
    assert reg.status_code == 201
    return reg.json()["access_token"]


def test_chat_persistence_and_history(client, test_user_token, test_db_session):
    """Test POST /v1/chat persists messages and GET /v1/chat/history retrieves them."""
    # 1. Check history is initially empty
    resp = client.get(
        "/v1/chat/history",
        headers={"Authorization": f"Bearer {test_user_token}"},
    )
    assert resp.status_code == 200
    initial_history = resp.json()
    assert len(initial_history) == 0

    # 2. Post a user question
    resp = client.post(
        "/v1/chat",
        json={"message": "What should I eat for dinner?"},
        headers={"Authorization": f"Bearer {test_user_token}"},
    )
    assert resp.status_code == 200
    reply_data = resp.json()
    assert "reply" in reply_data
    assert "dinner" in reply_data["reply"].lower()

    # 3. Retrieve history and assert both user and assistant messages exist
    resp = client.get(
        "/v1/chat/history",
        headers={"Authorization": f"Bearer {test_user_token}"},
    )
    assert resp.status_code == 200
    history = resp.json()
    assert len(history) == 2

    user_entry = history[0]
    bot_entry = history[1]

    assert user_entry["role"] == "user"
    assert user_entry["message"] == "What should I eat for dinner?"
    assert bot_entry["role"] == "assistant"
    assert bot_entry["message"] == reply_data["reply"]


def test_dinner_gap_engine_deficit_and_swaps(client, test_user_token, test_db_session):
    """Verify dinner gap engine recommends protein/fiber swaps for remaining deficits."""
    user = test_db_session.query(User).first()
    today = date.today()

    engine = WeeklyDigestEngine(test_db_session)
    advice, swaps = engine.generate_dinner_recommendation(user.id, today)

    assert isinstance(advice, str)
    assert len(advice) > 0
    assert "dinner" in advice.lower()
    assert validate_non_shaming_text(advice)

    # Swaps should contain candidate foods
    assert isinstance(swaps, list)
    assert len(swaps) > 0
    for swap in swaps:
        assert validate_non_shaming_text(swap.rationale)


def test_daily_rollup_includes_dinner_advice(client, test_user_token):
    """GET /v1/days/{day_str} must return dinner_advice and dinner_swaps."""
    today_str = date.today().isoformat()
    resp = client.get(
        f"/v1/days/{today_str}",
        headers={"Authorization": f"Bearer {test_user_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "dinner_advice" in data
    assert data["dinner_advice"] is not None
    assert "dinner_swaps" in data
    assert isinstance(data["dinner_swaps"], list)
    assert validate_non_shaming_text(data["dinner_advice"])


def test_invariant_i8_non_shaming_copy_on_budget_exceeded(client, test_user_token, test_db_session):
    """When daily target calories are exceeded, dinner engine MUST NOT shame or mention 'over budget'."""
    user = test_db_session.query(User).first()
    today = date.today()

    # Log a large meal exceeding 2500 kcal
    large_meal = Meal(
        user_id=user.id,
        context="personal_home",
        meal_slot="lunch",
        status="analyzed",
        source="manual",
        captured_at=datetime.now(timezone.utc),
        local_date=today,
        kcal_mean=2800.0,
        kcal_sd=50.0,
        kcal_p10=2736.0,
        kcal_p90=2864.0,
    )
    test_db_session.add(large_meal)
    test_db_session.commit()

    engine = WeeklyDigestEngine(test_db_session)
    advice, swaps = engine.generate_dinner_recommendation(user.id, today)

    # Invariant I8 validation
    assert validate_non_shaming_text(advice)
    advice_lower = advice.lower()
    banned_words = ["over budget", "you failed", "streak lost", "cheat", "bad job", "guilt", "punish"]
    for banned in banned_words:
        assert banned not in advice_lower, f"Banned shaming word '{banned}' found in advice: {advice}"

    # Verify empowering tone
    assert any(w in advice_lower for w in ["satisfying", "energy", "hydration", "soup", "salad"])
