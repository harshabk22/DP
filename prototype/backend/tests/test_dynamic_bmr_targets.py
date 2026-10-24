"""Automated integration and unit tests for Chunk D: Dynamic Mifflin-St Jeor Targets & Invariant I8 BMR Floor.

Verifies:
1. Accurate Mifflin-St Jeor calculation for male and female profiles with anthropometrics.
2. Physical activity multipliers (sedentary, light, moderate, active, very_active) and goal deltas (maintain, lose, gain).
3. Strict Invariant I8 enforcement: Calorie target is clamped to >= BMR even if a deficit goal mathematically drops it below.
4. Robust fallback handling for None or missing profile fields (2000 kcal, 75g protein, 250g carb, 65g fat).
5. Integration with GET /v1/days/{today} verifying dynamic targets match user profile state.
6. Integration with WeeklyDigestEngine.generate_dinner_recommendation verifying dynamic targets.
7. Integration with POST /v1/chat verifying assistant guidance grounds in dynamic profile targets.
"""

from datetime import date, datetime, timezone
import math
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.database import SessionLocal, sync_engine
from app.core.safety import (
    ACTIVITY_MULTIPLIERS,
    GOAL_ADJUSTMENTS,
    calculate_bmr,
    clamp_goal_target,
    get_user_daily_targets,
    validate_non_shaming_text,
)
from app.main import app
from app.models.base import Base
from app.models.meal import Meal, MealItem
from app.models.triggers import register_i5_triggers
from app.models.user import Profile, User
from app.services.digest.engine import WeeklyDigestEngine
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


# ===========================================================================
# Unit Tests: Mifflin-St Jeor & Invariant I8 Clamping
# ===========================================================================

def test_mifflin_st_jeor_bmr_calculation_accuracy():
    """Verify exact mathematical fidelity of Mifflin-St Jeor BMR for male, female, and other."""
    curr_year = datetime.now(timezone.utc).year

    # Male: 75 kg, 180 cm, birth_year = curr_year - 30 (age 30)
    # Formula: 10 * 75 + 6.25 * 180 - 5 * 30 + 5 = 750 + 1125 - 150 + 5 = 1730.0
    bmr_m = calculate_bmr(
        birth_year=curr_year - 30,
        sex="male",
        weight_kg=75.0,
        height_cm=180.0,
        current_year=curr_year,
    )
    assert math.isclose(bmr_m, 1730.0, rel_tol=1e-3)

    # Female: 60 kg, 165 cm, birth_year = curr_year - 28 (age 28)
    # Formula: 10 * 60 + 6.25 * 165 - 5 * 28 - 161 = 600 + 1031.25 - 140 - 161 = 1330.25 -> 1330.2
    bmr_f = calculate_bmr(
        birth_year=curr_year - 28,
        sex="female",
        weight_kg=60.0,
        height_cm=165.0,
        current_year=curr_year,
    )
    assert math.isclose(bmr_f, 1330.2, rel_tol=1e-2)

    # Missing physical metrics: WHO adult reference fallbacks
    # Male: 70 kg, 175 cm, age 30 -> 10*70 + 6.25*175 - 5*30 + 5 = 700 + 1093.75 - 150 + 5 = 1648.75 -> 1648.8
    bmr_fallback_m = calculate_bmr(birth_year=curr_year - 30, sex="male", current_year=curr_year)
    assert math.isclose(bmr_fallback_m, 1648.8, rel_tol=1e-2)


def test_activity_multipliers_and_goal_adjustments():
    """Verify PAL multipliers and caloric deficit/surplus adjustments."""
    curr_year = datetime.now(timezone.utc).year
    birth_year = curr_year - 30  # age 30

    # Male: 75 kg, 180 cm -> BMR = 1730.0
    profile_base = {
        "birth_year": birth_year,
        "sex": "male",
        "weight_kg": 75.0,
        "height_cm": 180.0,
    }

    # 1. Sedentary (1.2) + Maintain: 1730 * 1.2 = 2076.0
    target_sed, _, _, _ = get_user_daily_targets({**profile_base, "activity_level": "sedentary", "goal": "maintain"})
    assert math.isclose(target_sed, 2076.0, rel_tol=1e-2)

    # 2. Light (1.375) + Maintain: 1730 * 1.375 = 2378.75 -> 2378.8
    target_light, _, _, _ = get_user_daily_targets({**profile_base, "activity_level": "light", "goal": "maintain"})
    assert math.isclose(target_light, 2378.8, rel_tol=1e-2)

    # 3. Moderate (1.55) + Maintain: 1730 * 1.55 = 2681.5
    target_mod, _, _, _ = get_user_daily_targets({**profile_base, "activity_level": "moderate", "goal": "maintain"})
    assert math.isclose(target_mod, 2681.5, rel_tol=1e-2)

    # 4. Moderate (1.55) + Gain (+500): 2681.5 + 500 = 3181.5
    target_gain, _, _, _ = get_user_daily_targets({**profile_base, "activity_level": "moderate", "goal": "gain"})
    assert math.isclose(target_gain, 3181.5, rel_tol=1e-2)

    # 5. Active (1.725) + Lose (-500): 1730 * 1.725 - 500 = 2984.25 - 500 = 2484.25 -> 2484.2
    target_active_lose, _, _, _ = get_user_daily_targets({**profile_base, "activity_level": "active", "goal": "lose"})
    assert math.isclose(target_active_lose, 2484.2, rel_tol=1e-2)


def test_invariant_i8_bmr_floor_strict_enforcement():
    """Strict Invariant I8: Target is clamped to >= BMR even if deficit goal would mathematically drop it below."""
    curr_year = datetime.now(timezone.utc).year

    # Female: 48 kg, 152 cm, age 35 (birth_year = curr_year - 35)
    # BMR: 10*48 + 6.25*152 - 5*35 - 161 = 480 + 950 - 175 - 161 = 1094.0 kcal
    # Sedentary TDEE: 1094.0 * 1.2 = 1312.8 kcal
    # Goal 'lose' (-500 kcal): 1312.8 - 500 = 812.8 kcal (DANGEROUS CRASH DIET!)
    # Invariant I8 MUST CLAMP target to BMR floor = 1094.0 kcal!
    profile_crash = {
        "birth_year": curr_year - 35,
        "sex": "female",
        "weight_kg": 48.0,
        "height_cm": 152.0,
        "activity_level": "sedentary",
        "goal": "lose",
    }

    target_kcal, target_p, target_c, target_f = get_user_daily_targets(profile_crash)
    bmr_expected = 1094.0

    # Strict invariant assertion: target must equal BMR, never the unhealthily low 812.8 kcal
    assert target_kcal >= bmr_expected
    assert math.isclose(target_kcal, bmr_expected, rel_tol=1e-2)

    # Explicit override below BMR must also be clamped
    profile_override = {**profile_crash, "target_kcal": 750.0}
    target_override, _, _, _ = get_user_daily_targets(profile_override)
    assert target_override == bmr_expected


def test_macros_derivation_and_caloric_balance():
    """Verify macro derivation: protein (1.6g/kg), fat (25% kcal / 9), and carbs (remaining kcal / 4)."""
    curr_year = datetime.now(timezone.utc).year
    profile = {
        "birth_year": curr_year - 26,
        "sex": "male",
        "weight_kg": 70.0,
        "height_cm": 175.0,
        "activity_level": "moderate",
        "goal": "maintain",
    }
    target_kcal, target_p, target_c, target_f = get_user_daily_targets(profile)

    # Protein: 70 * 1.6 = 112.0g
    assert target_p == 112.0

    # Fat: 25% of target_kcal / 9
    expected_fat = round((target_kcal * 0.25) / 9.0, 1)
    assert target_f == expected_fat

    # Carbs: remaining kcal / 4
    expected_carbs = round((target_kcal - (112.0 * 4.0) - (expected_fat * 9.0)) / 4.0, 1)
    assert target_c == expected_carbs

    # Sum of macro energy aligns with total target calories within rounding
    macro_energy = (target_p * 4.0) + (target_f * 9.0) + (target_c * 4.0)
    assert math.isclose(macro_energy, target_kcal, abs_tol=5.0)


def test_none_and_empty_profile_defaults():
    """Verify that None or incomplete profiles return sensible standard adult defaults."""
    default_target = (2000.0, 75.0, 250.0, 65.0)

    assert get_user_daily_targets(None) == default_target
    assert get_user_daily_targets({}) == default_target
    assert get_user_daily_targets({"birth_year": None}) == default_target


# ===========================================================================
# Integration Tests: Endpoints & Services
# ===========================================================================

def test_daily_rollup_endpoint_returns_dynamic_targets(client, test_db_session):
    """Verify GET /v1/days/{today} returns dynamic target calories and macros matching profile anthropometrics."""
    # 1. Register a male user
    curr_year = datetime.now(timezone.utc).year
    email = f"dynamic_rollup_{datetime.now(timezone.utc).timestamp()}@example.com"
    reg = client.post(
        "/v1/auth/register",
        json={"email": email, "password": "Password123!", "birth_year": curr_year - 25, "sex": "male"},
    )
    assert reg.status_code == 201
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Update profile with specific metrics: 80 kg, 182 cm, active, maintain
    update_res = client.put(
        "/v1/me/profile",
        headers=headers,
        json={
            "weight_kg": 80.0,
            "height_cm": 182.0,
            "activity_level": "active",
            "goal": "maintain",
        },
    )
    assert update_res.status_code == 200
    prof_data = update_res.json()
    expected_bmr = prof_data["bmr"]

    # Expected dynamic target
    mock_prof = {
        "birth_year": curr_year - 25,
        "sex": "male",
        "weight_kg": 80.0,
        "height_cm": 182.0,
        "activity_level": "active",
        "goal": "maintain",
    }
    exp_kcal, exp_p, exp_c, exp_f = get_user_daily_targets(mock_prof)

    # 3. Call GET /v1/days/today
    today_str = date.today().isoformat()
    day_res = client.get(f"/v1/days/{today_str}", headers=headers)
    assert day_res.status_code == 200
    day_data = day_res.json()

    # Dynamic target assertions
    assert day_data["target_kcal"] == exp_kcal
    assert day_data["target_protein_g"] == exp_p
    assert day_data["target_carb_g"] == exp_c
    assert day_data["target_fat_g"] == exp_f
    assert day_data["remaining_budget_kcal"] == exp_kcal  # 0 meals logged so far
    assert day_data["target_kcal"] > 2000.0  # Dynamic target reflects active male metrics!


def test_daily_rollup_enforces_i8_bmr_floor_integration(client, test_db_session):
    """Verify GET /v1/days/{today} clamps target to BMR even if profile has steep deficit."""
    curr_year = datetime.now(timezone.utc).year
    email = f"i8_clamp_user_{datetime.now(timezone.utc).timestamp()}@example.com"
    reg = client.post(
        "/v1/auth/register",
        json={"email": email, "password": "Password123!", "birth_year": curr_year - 30, "sex": "female"},
    )
    assert reg.status_code == 201
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Set light petite female profile with 'lose' goal
    client.put(
        "/v1/me/profile",
        headers=headers,
        json={
            "weight_kg": 46.0,
            "height_cm": 150.0,
            "activity_level": "sedentary",
            "goal": "lose",
        },
    )

    today_str = date.today().isoformat()
    day_res = client.get(f"/v1/days/{today_str}", headers=headers)
    assert day_res.status_code == 200
    day_data = day_res.json()

    # Female BMR: 10*46 + 6.25*150 - 5*30 - 161 = 460 + 937.5 - 150 - 161 = 1086.5 kcal
    # Target kcal must be clamped to >= BMR
    assert day_data["target_kcal"] >= 1086.0
    assert day_data["target_kcal"] == 1086.5


def test_dinner_gap_engine_uses_dynamic_targets(client, test_db_session):
    """Verify WeeklyDigestEngine.generate_dinner_recommendation reflects user's dynamic BMR target."""
    curr_year = datetime.now(timezone.utc).year
    email = f"dinner_engine_{datetime.now(timezone.utc).timestamp()}@example.com"
    reg = client.post(
        "/v1/auth/register",
        json={"email": email, "password": "Password123!", "birth_year": curr_year - 24, "sex": "male"},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Active user with 85kg -> higher caloric budget (~2800 kcal)
    client.put(
        "/v1/me/profile",
        headers=headers,
        json={
            "weight_kg": 85.0,
            "height_cm": 185.0,
            "activity_level": "active",
            "goal": "maintain",
        },
    )

    user = test_db_session.query(User).filter(User.email == email).first()
    engine = WeeklyDigestEngine(test_db_session)
    advice, swaps = engine.generate_dinner_recommendation(user.id, date.today())

    # Advice must be non-shaming and cite dynamic remaining calories
    assert validate_non_shaming_text(advice)
    # Remaining kcal should reflect the higher target (~2800+), not 2000
    mock_prof = {
        "birth_year": curr_year - 24,
        "sex": "male",
        "weight_kg": 85.0,
        "height_cm": 185.0,
        "activity_level": "active",
        "goal": "maintain",
    }
    target_kcal, _, _, _ = get_user_daily_targets(mock_prof)
    assert str(round(target_kcal)) in advice


def test_copilot_chat_grounded_in_dynamic_targets(client, test_db_session):
    """Verify POST /v1/chat assistant queries report user's dynamic targets and protein goals."""
    curr_year = datetime.now(timezone.utc).year
    email = f"copilot_dyn_{datetime.now(timezone.utc).timestamp()}@example.com"
    reg = client.post(
        "/v1/auth/register",
        json={"email": email, "password": "Password123!", "birth_year": curr_year - 25, "sex": "male"},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Profile: 80 kg -> protein goal = 80 * 1.6 = 128.0g
    client.put(
        "/v1/me/profile",
        headers=headers,
        json={
            "weight_kg": 80.0,
            "height_cm": 178.0,
            "activity_level": "moderate",
            "goal": "maintain",
        },
    )

    # Chat query: "How much protein do I have left today?"
    chat_res = client.post(
        "/v1/chat",
        headers=headers,
        json={"message": "How much protein do I have left today?"},
    )
    assert chat_res.status_code == 200
    data = chat_res.json()

    # Macro overview in chat response matches dynamic target
    macros = data["macros"]
    assert macros["protein_g"] == 0.0
    assert "128.0g goal" in data["reply"]
    assert validate_non_shaming_text(data["reply"])
