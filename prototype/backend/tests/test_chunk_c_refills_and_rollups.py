"""Automated integration tests for Chunk C: 1-Tap Refills, Recalculations & Daily Rollup Contracts.

Verifies:
1. PATCH /v1/meal-items/{item_id} updating unit count and is_refill=True:
   - Appends immutable correction to audit log (Invariant I5).
   - Recalculates meal moments (mean, sd, p10, p90) with closed-form analytical engine.
2. GET /v1/days/{today} returns dinner_advice, dinner_swaps, and canonical total_carb_g.
3. Invariant I8 non-shaming compliance for dinner advice and swap recommendations.
4. Route prefix parity for /api/v1/meal-items and /api/v1/days.
"""

from datetime import date, datetime, timezone
import math
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.database import SessionLocal, sync_engine
from app.core.safety import validate_non_shaming_text
from app.main import app
from app.models.base import Base
from app.models.meal import Meal, MealItem
from app.models.question import Correction
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
    email = f"refill_tester_{datetime.now(timezone.utc).timestamp()}@example.com"
    pwd = "SecurePassword123!"
    reg_res = client.post(
        "/v1/auth/register",
        json={"email": email, "password": pwd, "birth_year": 1996, "sex": "male", "goal": "maintain"},
    )
    assert reg_res.status_code == 201
    token = reg_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_meal_item_refill_patch_analytical_recalculation(client, auth_headers, test_db_session):
    """Verify PATCH /v1/meal-items/{item_id} updates unit count, is_refill=True, and recalculates meal totals."""
    # 1. Log a meal with 2 items via quick-log text
    log_res = client.post(
        "/v1/meals/text",
        headers=auth_headers,
        json={"text": "2 rotis and 1 bowl dal tadka", "meal_slot": "lunch", "context": "personal_home"},
    )
    assert log_res.status_code == 201
    meal_data = log_res.json()
    meal_id = meal_data["meal_id"]
    assert len(meal_data["items"]) >= 2

    # Find the roti item
    target_item = None
    for item in meal_data["items"]:
        if "roti" in item["dish_label"].lower():
            target_item = item
            break
    if not target_item:
        target_item = meal_data["items"][0]

    item_id = target_item["id"]
    initial_count = target_item["unit_count"]
    initial_item_kcal = target_item["kcal_mean"]
    initial_meal_kcal = meal_data["total"]["kcal_mean"]
    initial_meal_sd = meal_data["total"]["kcal_sd"]

    # 2. Tap 1-Tap Refill Chip (+1 unit, is_refill=True)
    patch_res = client.patch(
        f"/v1/meal-items/{item_id}",
        headers=auth_headers,
        json={"unit_count": initial_count + 1.0, "is_refill": True},
    )
    assert patch_res.status_code == 200
    patched = patch_res.json()

    # Verify item-level analytical updates
    assert patched["is_refill"] is True
    assert patched["unit_count"] == initial_count + 1.0
    assert patched["kcal_mean"] > initial_item_kcal
    assert patched["grams_mean"] > target_item["grams_mean"]

    # 3. Query parent meal to verify analytical closed-form recalculation (p01.md §4.1)
    meal_res = client.get(f"/v1/meals/{meal_id}", headers=auth_headers)
    assert meal_res.status_code == 200
    updated_meal = meal_res.json()

    assert updated_meal["total"]["kcal_mean"] > initial_meal_kcal
    assert updated_meal["total"]["kcal_p10"] < updated_meal["total"]["kcal_p90"]
    # Verify the updated item inside the meal has is_refill=True
    updated_items = {it["id"]: it for it in updated_meal["items"]}
    assert updated_items[item_id]["is_refill"] is True
    assert updated_items[item_id]["unit_count"] == initial_count + 1.0

    # 4. Verify Invariant I5: Immutable audit correction created
    correction = (
        test_db_session.query(Correction)
        .filter(Correction.meal_item_id == uuid.UUID(item_id))
        .first()
    )
    assert correction is not None
    assert correction.source == "manual_edit"
    assert correction.new_value.get("is_refill") is True


def test_daily_rollup_contract_carbs_and_dinner_guidance(client, auth_headers):
    """Verify GET /v1/days/{today} returns total_carb_g (not carbs), dinner_advice, and dinner_swaps."""
    # 1. Log a meal for today
    log_res = client.post(
        "/v1/meals/text",
        headers=auth_headers,
        json={"text": "1 plate rice and 1 bowl rajma", "meal_slot": "lunch", "context": "personal_home"},
    )
    assert log_res.status_code == 201

    # 2. Fetch daily rollup via canonical path /v1/days/today
    today_res = client.get("/v1/days/today", headers=auth_headers)
    assert today_res.status_code == 200
    rollup = today_res.json()

    # Contract verification: total_carb_g key exists and is non-negative float
    assert "total_carb_g" in rollup
    assert isinstance(rollup["total_carb_g"], (int, float))
    assert rollup["total_carb_g"] > 0.0

    # Verify other macro fields
    assert "total_protein_g" in rollup
    assert "total_fat_g" in rollup
    assert "total_fibre_g" in rollup
    assert "total_kcal_mean" in rollup
    assert rollup["total_protein_g"] > 0.0

    # Proactive 19:00 dinner gap engine verification
    assert "dinner_advice" in rollup
    assert rollup["dinner_advice"] is not None
    assert len(rollup["dinner_advice"]) > 10

    # Invariant I8: Ensure dinner advice contains zero shaming copy
    assert validate_non_shaming_text(rollup["dinner_advice"])

    # Dinner food swaps verification
    assert "dinner_swaps" in rollup
    assert isinstance(rollup["dinner_swaps"], list)
    if rollup["dinner_swaps"]:
        swap = rollup["dinner_swaps"][0]
        assert "food_id" in swap
        assert "name" in swap
        assert "target_nutrient" in swap
        assert "nutrient_density" in swap
        assert "serving_description" in swap
        assert "rationale" in swap
        assert validate_non_shaming_text(swap["rationale"])

    # Check meals list in rollup
    assert len(rollup["meals"]) >= 1
    meal_entry = rollup["meals"][0]
    assert "meal_id" in meal_entry
    assert "items" in meal_entry
    assert "total" in meal_entry


def test_refill_via_api_prefix_and_date_variations(client, auth_headers):
    """Verify route prefix parity for /api/v1/meal-items and explicit date rollups."""
    # Log meal
    log_res = client.post(
        "/api/v1/meals/text",
        headers=auth_headers,
        json={"text": "1 cup curd yogurt and 1 apple", "meal_slot": "breakfast"},
    )
    assert log_res.status_code == 201
    meal_data = log_res.json()
    item_id = meal_data["items"][0]["id"]

    # PATCH via /api/v1 prefix
    patch_res = client.patch(
        f"/api/v1/meal-items/{item_id}",
        headers=auth_headers,
        json={"unit_count": 2.0, "is_refill": True},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["is_refill"] is True

    # GET /api/v1/days/{today_iso}
    today_iso = date.today().isoformat()
    rollup_res = client.get(f"/api/v1/days/{today_iso}", headers=auth_headers)
    assert rollup_res.status_code == 200
    data = rollup_res.json()
    assert data["date"] == today_iso
    assert "total_carb_g" in data
    assert data["dinner_advice"] is not None
