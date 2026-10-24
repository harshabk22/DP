"""Unit and API integration tests for Human-First AI Nutrition Assistant features.

Validates:
1. Natural language text & voice transcript quick-logging (POST /v1/meals/text).
2. Conversational Nutrition Co-Pilot Q&A grounding (POST /v1/chat).
3. Text meal parsing across colloquial and multi-item dietary inputs.
4. Exact analytical calorie & macro calculation with honest [p10, p90] ranges.
"""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.main import app
from app.models.base import Base
from app.models.triggers import register_i5_triggers
from app.services.nutrition.text_parser import TextMealParser
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
    email = f"assistant_tester_{datetime.now(timezone.utc).timestamp()}@example.com"
    pwd = "StrongPassword123!"
    reg = client.post(
        "/v1/auth/register",
        json={"email": email, "password": pwd, "birth_year": 1998, "sex": "female"},
    )
    assert reg.status_code == 201
    return reg.json()["access_token"]


def test_text_meal_parser_colloquial_phrases():
    """Verify TextMealParser correctly extracts dishes, units, and counts."""
    # Case 1: Indian staple
    items = TextMealParser.parse("had 2 rotis and 1 bowl dal tadka")
    assert len(items) == 2
    assert items[0].dish_label == "roti"
    assert items[0].unit_count == 2.0
    assert items[0].unit == "slice_piece"
    assert items[1].dish_label == "dal tadka"
    assert items[1].unit_count == 1.0

    # Case 2: Snack / beverage
    items2 = TextMealParser.parse("one cup curd and an apple")
    assert len(items2) == 2
    assert items2[0].dish_label == "curd"
    assert items2[0].unit_count == 1.0
    assert items2[1].dish_label == "apple"
    assert items2[1].unit_count == 1.0

    # Case 3: Eggs and bread
    items3 = TextMealParser.parse("2 eggs and 2 slices bread")
    assert len(items3) == 2
    assert items3[0].dish_label == "egg"
    assert items3[0].unit_count == 2.0
    assert items3[1].dish_label == "bread"
    assert items3[1].unit_count == 2.0


def test_api_log_meal_via_text(client, test_user_token):
    """Test POST /v1/meals/text instantly parses, resolves, and logs a meal."""
    payload = {
        "text": "had 2 rotis and 1 bowl dal tadka",
        "meal_slot": "lunch",
        "context": "personal_home",
        "source": "text",
    }
    resp = client.post(
        "/v1/meals/text",
        json=payload,
        headers={"Authorization": f"Bearer {test_user_token}"},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()

    assert data["status"] == "analyzed"
    assert len(data["items"]) >= 2
    assert data["total"] is not None
    assert data["total"]["kcal_mean"] > 0
    assert data["total"]["kcal_p10"] <= data["total"]["kcal_mean"] <= data["total"]["kcal_p90"]
    assert data["total"]["protein_g"] >= 0.0


def test_api_conversational_copilot_protein_and_dinner(client, test_user_token):
    """Test POST /v1/chat provides grounded, encouraging responses based on daily logs."""
    # 1. Ask about protein when 0 logged
    resp1 = client.post(
        "/v1/chat",
        json={"message": "How much protein do I have left today?"},
        headers={"Authorization": f"Bearer {test_user_token}"},
    )
    assert resp1.status_code == 200, resp1.text
    data1 = resp1.json()
    assert "protein" in data1["reply"].lower()
    assert data1["macros"]["consumed_kcal"] == 0.0
    assert data1["macros"]["remaining_kcal"] > 0

    # 2. Log a meal via text first
    meal_resp = client.post(
        "/v1/meals/text",
        json={"text": "2 rotis and 1 bowl dal tadka", "meal_slot": "lunch"},
        headers={"Authorization": f"Bearer {test_user_token}"},
    )
    assert meal_resp.status_code == 201

    # 3. Ask about dinner suggestions
    resp2 = client.post(
        "/v1/chat",
        json={"message": "What should I eat for dinner?"},
        headers={"Authorization": f"Bearer {test_user_token}"},
    )
    assert resp2.status_code == 200, resp2.text
    data2 = resp2.json()
    assert "dinner" in data2["reply"].lower()
    assert data2["macros"]["consumed_kcal"] > 0
    assert data2["macros"]["remaining_kcal"] >= 0
