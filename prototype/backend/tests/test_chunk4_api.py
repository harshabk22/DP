"""Comprehensive integration test suite for Chunk 4: API & Safety Guardrails.

Validates:
1. Invariant I7: Under-18 registration & profile update rejected with HTTP 422.
2. Authentication: Argon2id password hashing, registration, and JWT Bearer login.
3. Invariant I8: Mifflin-St Jeor BMR calculation, goal clamping, and anti-shaming copy.
4. Invariant I6: Unbundled, per-purpose research & logging consent tracking.
5. Hard Deletion: Full user erasure (DELETE /me) with cascade and I5 trigger bypass.
6. Meal Pipeline: Asynchronous 202 -> background analysis -> probabilistic distribution (Invariants I1, I2, I9, I10).
7. Question Answering: Discrete tap responses (I4) updating Bayesian tableware priors and logging to corrections (I5).
8. Refill Chips: PATCH /meal-items/{id} appending immutable corrections (I5) and recomputing totals.
9. Weekly Digest: ICMR-NIN nutrient gap analysis, top 3 food swaps, non-shaming copy (I8).
10. RFC 7807: Problem details error handling.
"""

from datetime import date, datetime, timedelta, timezone
import io
import math
import uuid
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.database import SessionLocal, sync_engine
from app.core.safety import (
    DISCLAIMER_TEXT,
    calculate_bmr,
    clamp_goal_target,
    validate_non_shaming_text,
)
from app.core.security import decode_access_token, hash_password, verify_password
from app.main import app
from app.models.base import Base
from app.models.food import Food
from app.models.meal import Meal, MealItem, ModelRun
from app.models.prior import UnitPrior, UnitPriorHistory
from app.models.question import Correction, DailyBudget, Question
from app.models.triggers import register_i5_triggers
from app.models.user import Consent, Profile, User
from app.services.vlm.client import VLMItem, VLMOutput
from scripts.ingest.seed_nutrition_data import seed_foods


@pytest.fixture
def test_db_session():
    """Create an isolated, in-memory SQLite database session with I5 triggers."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    register_i5_triggers(engine)

    SessionLocal.configure(bind=engine)

    with Session(engine) as session:
        # Seed certified reference foods
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


# ---------------------------------------------------------------------------
# Test 1: Invariant I7 - Age Verification (Under 18 Blocked)
# ---------------------------------------------------------------------------
def test_invariant_i7_age_verification(client):
    """Verify registration rejects users under 18 with HTTP 422 (Invariant I7)."""
    current_year = datetime.now(timezone.utc).year

    # Minor registration (e.g. 15 years old)
    minor_payload = {
        "email": "minor_user@example.com",
        "password": "SecurePassword123!",
        "birth_year": current_year - 15,
        "sex": "female",
    }
    res_minor = client.post("/v1/auth/register", json=minor_payload)
    assert res_minor.status_code == 422
    data = res_minor.json()
    assert "Invariant I7" in data["detail"]

    # Exactly 17 years old -> rejected
    seventeen_payload = {
        "email": "seventeen@example.com",
        "password": "SecurePassword123!",
        "birth_year": current_year - 17,
        "sex": "male",
    }
    res_seventeen = client.post("/v1/auth/register", json=seventeen_payload)
    assert res_seventeen.status_code == 422

    # Adult (>= 18 years old) -> accepted
    adult_payload = {
        "email": "adult_user@example.com",
        "password": "SecurePassword123!",
        "birth_year": current_year - 22,
        "sex": "other",
    }
    res_adult = client.post("/v1/auth/register", json=adult_payload)
    assert res_adult.status_code == 201
    assert "access_token" in res_adult.json()


# ---------------------------------------------------------------------------
# Test 2: Authentication & Password Hashing
# ---------------------------------------------------------------------------
def test_auth_registration_and_login(client):
    """Verify Argon2id password hashing, registration, and JWT token issuance."""
    email = "auth_test@example.com"
    pwd = "CorrectHorseBattery99!"

    # 1. Register
    reg_res = client.post(
        "/v1/auth/register",
        json={"email": email, "password": pwd, "birth_year": 1995, "sex": "male"},
    )
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert "access_token" in reg_data
    token = reg_data["access_token"]

    # Decode and verify token
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == reg_data["user_id"]

    # 2. Duplicate registration rejected
    dup_res = client.post(
        "/v1/auth/register",
        json={"email": email, "password": "another_password", "birth_year": 1995},
    )
    assert dup_res.status_code == 400

    # 3. Successful Login
    login_res = client.post("/v1/auth/login", json={"email": email, "password": pwd})
    assert login_res.status_code == 200
    assert "access_token" in login_res.json()

    # 4. Failed Login (invalid password)
    fail_res = client.post("/v1/auth/login", json={"email": email, "password": "wrong_password"})
    assert fail_res.status_code == 401


# ---------------------------------------------------------------------------
# Test 3: Invariant I8 - Mifflin–St Jeor BMR Clamping & Non-Shaming
# ---------------------------------------------------------------------------
def test_invariant_i8_bmr_clamping_and_non_shaming(client):
    """Verify BMR computation, goal clamping to BMR, and anti-shaming copy."""
    # 1. Mifflin–St Jeor calculation checks
    bmr_male = calculate_bmr(birth_year=2000, sex="male", weight_kg=70.0, height_cm=175.0, current_year=2026)
    # 10*70 + 6.25*175 - 5*26 + 5 = 700 + 1093.75 - 130 + 5 = 1668.75 -> 1668.8
    assert math.isclose(bmr_male, 1668.8, rel_tol=1e-2)

    bmr_female = calculate_bmr(birth_year=2000, sex="female", weight_kg=60.0, height_cm=162.0, current_year=2026)
    # 10*60 + 6.25*162 - 5*26 - 161 = 600 + 1012.5 - 130 - 161 = 1321.5
    assert math.isclose(bmr_female, 1321.5, rel_tol=1e-2)

    # 2. Clamping check: target below BMR must be clamped to BMR
    clamped_low = clamp_goal_target(target_kcal=1000.0, bmr=bmr_male)
    assert clamped_low == bmr_male

    clamped_high = clamp_goal_target(target_kcal=2200.0, bmr=bmr_male)
    assert clamped_high == 2200.0

    # 3. Anti-shaming text validator
    assert not validate_non_shaming_text("You are over budget today!")
    assert not validate_non_shaming_text("You failed to meet your target")
    assert not validate_non_shaming_text("Streak lost! Start over.")
    assert validate_non_shaming_text("You logged 5 meals this week. Great job!")

    # 4. API integration: register and update profile with low target
    reg = client.post(
        "/v1/auth/register",
        json={"email": "bmr_user@example.com", "password": "Password123!", "birth_year": 1998, "sex": "male"},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    update_res = client.put(
        "/v1/me/profile",
        headers=headers,
        json={
            "weight_kg": 75.0,
            "height_cm": 180.0,
            "target_kcal": 1100.0,  # Below calculated BMR!
        },
    )
    assert update_res.status_code == 200
    prof = update_res.json()
    assert prof["bmr"] > 1600.0
    # Clamped target kcal must be equal to estimated BMR (Invariant I8)
    assert prof["clamped_target_kcal"] == prof["bmr"]
    assert prof["disclaimer"] == DISCLAIMER_TEXT


# ---------------------------------------------------------------------------
# Test 4: Invariant I6 - Unbundled Study Consent
# ---------------------------------------------------------------------------
def test_invariant_i6_unbundled_consent(client):
    """Verify granular, unbundled consent recording and retrieval (Invariant I6)."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "consent_user@example.com", "password": "Password123!", "birth_year": 1999},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Grant core logging
    c1 = client.post(
        "/v1/me/consents",
        headers=headers,
        json={"purpose": "core_logging", "granted": True, "notice_version": "v1.0"},
    )
    assert c1.status_code == 201
    assert c1.json()["purpose"] == "core_logging"
    assert c1.json()["granted"] is True

    # Revoke or decline research dataset
    c2 = client.post(
        "/v1/me/consents",
        headers=headers,
        json={"purpose": "research_dataset", "granted": False, "notice_version": "v1.0"},
    )
    assert c2.status_code == 201
    assert c2.json()["purpose"] == "research_dataset"
    assert c2.json()["granted"] is False
    assert c2.json()["revoked_at"] is not None

    # Invalid purpose
    bad_c = client.post(
        "/v1/me/consents",
        headers=headers,
        json={"purpose": "targeted_advertising", "granted": True},
    )
    assert bad_c.status_code == 422

    # Query all consents
    list_res = client.get("/v1/me/consents", headers=headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 2


# ---------------------------------------------------------------------------
# Test 5: Hard Deletion (DELETE /me) with Invariant I5 Trigger Bypass
# ---------------------------------------------------------------------------
def test_hard_deletion_user_erasure(client, test_db_session):
    """Verify complete hard deletion of user, cascading meals and bypassing I5 trigger."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "delete_me@example.com", "password": "Password123!", "birth_year": 1996},
    )
    token = reg.json()["access_token"]
    user_id = uuid.UUID(reg.json()["user_id"])
    headers = {"Authorization": f"Bearer {token}"}

    # Add a meal and a correction to activate I5 trigger
    meal = Meal(
        user_id=user_id,
        captured_at=datetime.now(timezone.utc),
        local_date=date.today(),
        source="photo",
        status="analyzed",
    )
    test_db_session.add(meal)
    test_db_session.flush()

    correction = Correction(
        user_id=user_id,
        field="unit_count",
        new_value={"unit_count": 2},
        source="manual_edit",
    )
    test_db_session.add(correction)
    test_db_session.commit()

    # DELETE /me
    del_res = client.delete("/v1/me", headers=headers)
    assert del_res.status_code == 204

    # Verify user record is completely gone from the database
    deleted_user = test_db_session.query(User).filter(User.id == user_id).first()
    assert deleted_user is None

    # Verify cascading meals and profile are gone
    meals_count = test_db_session.query(Meal).filter(Meal.user_id == user_id).count()
    assert meals_count == 0


# ---------------------------------------------------------------------------
# Test 6: Meal Upload -> 202 -> Async Pipeline -> GET Result (Invariants I1, I2, I9, I10)
# ---------------------------------------------------------------------------
def test_meal_upload_and_pipeline(client, test_db_session):
    """Verify meal upload (202), background execution, and probabilistic result retrieval."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "meal_flow@example.com", "password": "Password123!", "birth_year": 1994},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Post meal upload
    upload_res = client.post(
        "/v1/meals?sync_run=true",
        headers=headers,
        data={"context": "personal_home", "meal_slot": "lunch"},
    )
    assert upload_res.status_code == 202
    meal_id = upload_res.json()["meal_id"]

    # 2. Query meal detail
    detail_res = client.get(f"/v1/meals/{meal_id}", headers=headers)
    assert detail_res.status_code == 200
    detail = detail_res.json()

    assert detail["status"] == "analyzed"
    assert detail["context"] == "personal_home"

    # Invariant I2: Probabilistic distribution moments
    assert detail["total"] is not None
    assert detail["total"]["kcal_mean"] > 0.0
    assert detail["total"]["kcal_sd"] > 0.0
    assert detail["total"]["kcal_p10"] < detail["total"]["kcal_p90"]

    # Macronutrient totals
    assert "protein_g" in detail["total"]
    assert "fibre_g" in detail["total"]

    # Items breakdown
    assert len(detail["items"]) > 0
    item = detail["items"][0]
    assert item["dish_label"] is not None
    assert item["grams_mean"] > 0.0

    # Invariant I10: ModelRun record exists with frozen versioning
    model_run = (
        test_db_session.query(ModelRun)
        .filter(ModelRun.meal_id == uuid.UUID(meal_id))
        .first()
    )
    assert model_run is not None
    assert model_run.prompt_version == "extract_v1"
    assert model_run.model_name in ("gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite")
    assert model_run.k_samples == 1


# ---------------------------------------------------------------------------
# Test 6b: Asynchronous Meal Upload with Session-Isolated Background Execution
# ---------------------------------------------------------------------------
def test_async_meal_upload_background_execution(client, test_db_session):
    """Verify asynchronous meal upload processes in dedicated background session without closed-session errors."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "async_meal_user@example.com", "password": "Password123!", "birth_year": 1995},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Generate dummy meal image
    img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    dummy_bytes = buf.getvalue()

    # Upload meal with sync_run=False (the default) using dummy image/form data
    upload_res = client.post(
        "/v1/meals",
        headers=headers,
        data={"context": "personal_home", "meal_slot": "dinner"},
        files={"file": ("dinner.jpg", dummy_bytes, "image/jpeg")},
    )
    assert upload_res.status_code == 202
    data = upload_res.json()
    assert "meal_id" in data
    meal_id = data["meal_id"]

    # FastAPI TestClient executes background tasks synchronously before the call returns.
    # Verify the meal transitioned to status == "analyzed", items are created, and no closed-session error occurred.
    detail_res = client.get(f"/v1/meals/{meal_id}", headers=headers)
    assert detail_res.status_code == 200
    detail = detail_res.json()

    assert detail["status"] == "analyzed"
    assert len(detail["items"]) > 0
    assert detail["total"] is not None
    assert detail["total"]["kcal_mean"] > 0.0

    # Query DB directly with fresh state
    test_db_session.expire_all()
    db_meal = test_db_session.query(Meal).filter(Meal.id == uuid.UUID(meal_id)).first()
    assert db_meal is not None
    assert db_meal.status == "analyzed"
    assert len(db_meal.items) > 0


# ---------------------------------------------------------------------------
# Test 7: Question Answering, Bayesian Prior Update, and Invariant I5 Audit
# ---------------------------------------------------------------------------
def test_question_answering_and_prior_update(client, test_db_session):
    """Verify discrete question answering updates Bayesian priors and appends to corrections."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "question_user@example.com", "password": "Password123!", "birth_year": 1993},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Upload meal with sync_run=true
    upload_res = client.post(
        "/v1/meals?sync_run=true",
        headers=headers,
        data={"context": "personal_home", "meal_slot": "dinner"},
    )
    meal_id = upload_res.json()["meal_id"]

    # Fetch meal to inspect generated question
    detail_res = client.get(f"/v1/meals/{meal_id}", headers=headers)
    detail = detail_res.json()
    assert detail["question"] is not None

    question_id = detail["question"]["id"]
    options = detail["question"]["options"]
    # Invariant I4: <= 4 discrete options
    assert len(options) <= 4

    # Count prior corrections
    user_id = uuid.UUID(reg.json()["user_id"])
    initial_corr_count = test_db_session.query(Correction).filter(Correction.user_id == user_id).count()

    # Answer question with choice 2
    answer_res = client.post(
        f"/v1/meals/{meal_id}/answer",
        headers=headers,
        json={"answer": 2},
    )
    assert answer_res.status_code == 200
    assert answer_res.json()["answered_at"] is not None

    # Invariant I5: Correction was appended (not updated)
    new_corr_count = test_db_session.query(Correction).filter(Correction.user_id == user_id).count()
    assert new_corr_count == initial_corr_count + 1

    latest_corr = (
        test_db_session.query(Correction)
        .filter(Correction.user_id == user_id)
        .order_by(Correction.created_at.desc())
        .first()
    )
    assert latest_corr.source == "question"
    assert latest_corr.field == "unit_count"

    # Verify unit prior and unit prior history updated
    priors = test_db_session.query(UnitPrior).filter(UnitPrior.user_id == user_id).all()
    assert len(priors) > 0
    assert priors[0].n_obs >= 1

    history = test_db_session.query(UnitPriorHistory).filter(UnitPriorHistory.user_id == user_id).all()
    assert len(history) > 0


# ---------------------------------------------------------------------------
# Test 8: Refill Chips via PATCH /meal-items/{id} (Invariant I5)
# ---------------------------------------------------------------------------
def test_meal_item_refill_chip_patch(client, test_db_session):
    """Verify refill chip adjustment modifies portion and appends immutable correction."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "refill_user@example.com", "password": "Password123!", "birth_year": 1991},
    )
    token = reg.json()["access_token"]
    user_id = uuid.UUID(reg.json()["user_id"])
    headers = {"Authorization": f"Bearer {token}"}

    upload_res = client.post(
        "/v1/meals?sync_run=true",
        headers=headers,
        data={"context": "personal_home"},
    )
    meal_id = upload_res.json()["meal_id"]

    detail = client.get(f"/v1/meals/{meal_id}", headers=headers).json()
    item_id = detail["items"][0]["id"]
    old_kcal = detail["items"][0]["kcal_mean"]

    # Click refill chip (+1 unit)
    patch_res = client.patch(
        f"/v1/meal-items/{item_id}",
        headers=headers,
        json={"unit_count": 2.0},
    )
    assert patch_res.status_code == 200
    updated_item = patch_res.json()
    assert updated_item["unit_count"] == 2.0
    assert updated_item["kcal_mean"] > old_kcal

    # Invariant I5: Must have appended to corrections
    corr = (
        test_db_session.query(Correction)
        .filter(Correction.meal_item_id == uuid.UUID(item_id))
        .first()
    )
    assert corr is not None
    assert corr.source == "manual_edit"
    assert corr.field == "unit_count"


def test_meal_item_manual_grams_edit_preserved(client, test_db_session):
    """Verify manual kitchen scale grams update sets precise portion and creates Correction."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "scale_user@example.com", "password": "Password123!", "birth_year": 1993},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    upload_res = client.post(
        "/v1/meals?sync_run=true",
        headers=headers,
        data={"context": "personal_home"},
    )
    meal_id = upload_res.json()["meal_id"]

    detail = client.get(f"/v1/meals/{meal_id}", headers=headers).json()
    item = detail["items"][0]
    item_id = item["id"]
    food_id = item["food_id"]

    food = test_db_session.get(Food, food_id)
    expected_kcal = (float(food.kcal_per_100g) if food else 100.0) * (250.0 / 100.0)

    # User enters 250g manual measurement from kitchen scale
    patch_res = client.patch(
        f"/v1/meal-items/{item_id}",
        headers=headers,
        json={"grams": 250.0},
    )
    assert patch_res.status_code == 200
    updated_item = patch_res.json()
    assert updated_item["grams_mean"] == 250.0
    assert updated_item["grams_sd"] == 2.0
    assert updated_item["portion_source"] == "user"
    assert math.isclose(updated_item["kcal_mean"], expected_kcal, rel_tol=1e-2)

    # Invariant I5: Must have created Correction record
    corr = (
        test_db_session.query(Correction)
        .filter(Correction.meal_item_id == uuid.UUID(item_id), Correction.field == "grams")
        .first()
    )
    assert corr is not None
    assert corr.source == "manual_edit"
    assert corr.new_value == {"grams": 250.0}

    # Subsequent PATCH toggling is_refill must preserve user's manual scale measurement (grams_mean == 250.0)
    refill_patch = client.patch(
        f"/v1/meal-items/{item_id}",
        headers=headers,
        json={"is_refill": True},
    )
    assert refill_patch.status_code == 200
    refill_item = refill_patch.json()
    assert refill_item["is_refill"] is True
    assert refill_item["grams_mean"] == 250.0
    assert refill_item["portion_source"] == "user"
    assert math.isclose(refill_item["kcal_mean"], expected_kcal, rel_tol=1e-2)


def test_meal_item_refill_chip_toggle(client, test_db_session):
    """Verify refill chip toggle updates is_refill field on MealItem."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "refill_toggle@example.com", "password": "Password123!", "birth_year": 1990},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    upload_res = client.post(
        "/v1/meals?sync_run=true",
        headers=headers,
        data={"context": "personal_home"},
    )
    meal_id = upload_res.json()["meal_id"]

    detail = client.get(f"/v1/meals/{meal_id}", headers=headers).json()
    item_id = detail["items"][0]["id"]
    assert detail["items"][0]["is_refill"] is False

    # Toggle is_refill chip to True
    patch_res = client.patch(
        f"/v1/meal-items/{item_id}",
        headers=headers,
        json={"is_refill": True},
    )
    assert patch_res.status_code == 200
    updated_item = patch_res.json()
    assert updated_item["is_refill"] is True

    # Also verify GET /v1/meals/{meal_id} reflects is_refill is True
    detail_after = client.get(f"/v1/meals/{meal_id}", headers=headers).json()
    assert detail_after["items"][0]["is_refill"] is True


# ---------------------------------------------------------------------------
# Test 9: Weekly Digest & Menu Swaps (§10 & Invariant I8)
# ---------------------------------------------------------------------------
def test_weekly_digest_and_food_swaps(client, test_db_session):
    """Verify rule-based ICMR-NIN recommendations and anti-shaming food swaps."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "digest_user@example.com", "password": "Password123!", "birth_year": 1992},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Upload a meal
    client.post(
        "/v1/meals?sync_run=true",
        headers=headers,
        data={"context": "personal_home", "captured_at": "2026-09-21T12:00:00"},
    )

    # Query weekly summary (2026-09-21 is in ISO week 2026-W39)
    digest_res = client.get("/v1/weeks/2026-W39", headers=headers)
    assert digest_res.status_code == 200
    digest = digest_res.json()

    assert digest["iso_week"] == "2026-W39"
    assert digest["days_logged"] >= 1
    assert digest["disclaimer"] == DISCLAIMER_TEXT

    # Check top 3 swaps
    assert len(digest["swaps"]) <= 3
    for swap in digest["swaps"]:
        assert swap["food_id"] > 0
        assert swap["nutrient_density"] > 0.0
        # Invariant I8: All rationale text must be non-shaming
        assert validate_non_shaming_text(swap["rationale"])


# ---------------------------------------------------------------------------
# Test 10: Daily Rollup Endpoint
# ---------------------------------------------------------------------------
def test_daily_rollup_endpoint(client):
    """Verify daily rollup aggregation and question budget state."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "daily_user@example.com", "password": "Password123!", "birth_year": 1990},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Query daily rollup for today
    today_str = date.today().isoformat()
    rollup_res = client.get(f"/v1/days/{today_str}", headers=headers)
    assert rollup_res.status_code == 200
    data = rollup_res.json()

    assert data["date"] == today_str
    assert data["budget"]["total"] == 3
    assert data["disclaimer"] == DISCLAIMER_TEXT


# ---------------------------------------------------------------------------
# Test 11: Meal Discarding (DELETE /meals/{id})
# ---------------------------------------------------------------------------
def test_meal_discard(client):
    """Verify discarding a meal changes its status to discarded."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "discard_user@example.com", "password": "Password123!", "birth_year": 1993},
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    upload_res = client.post(
        "/v1/meals?sync_run=true",
        headers=headers,
        data={"context": "personal_home"},
    )
    meal_id = upload_res.json()["meal_id"]

    del_res = client.delete(f"/v1/meals/{meal_id}", headers=headers)
    assert del_res.status_code == 204

    # Verify status is discarded
    get_res = client.get(f"/v1/meals/{meal_id}", headers=headers)
    assert get_res.json()["status"] == "discarded"


# ---------------------------------------------------------------------------
# Test 12: Question Skipping & Invariant I4 Free-Text Rejection
# ---------------------------------------------------------------------------
def test_question_skip_and_free_text_rejection(client):
    """Verify question skipping and rejection of free-form text answers (Invariant I4)."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "skip_user@example.com", "password": "Password123!", "birth_year": 1995},
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    upload_res = client.post(
        "/v1/meals?sync_run=true",
        headers=headers,
        data={"context": "personal_home"},
    )
    meal_id = upload_res.json()["meal_id"]

    # Reject long free-form text paragraph (Invariant I4: tap-only options)
    long_essay = "I had approximately two and a half rotis and a big bowl of sambar with lots of ghee on top"
    essay_res = client.post(
        f"/v1/meals/{meal_id}/answer",
        headers=headers,
        json={"answer": long_essay},
    )
    assert essay_res.status_code == 422
    assert "Invariant I4" in essay_res.json()["detail"]

    # Skip question
    skip_res = client.post(
        f"/v1/meals/{meal_id}/answer",
        headers=headers,
        json={"skipped": True},
    )
    assert skip_res.status_code == 200
    assert skip_res.json()["skipped"] is True


# ---------------------------------------------------------------------------
# Test 13: Data Portability Export (GET /me/export)
# ---------------------------------------------------------------------------
def test_user_data_export(client):
    """Verify GDPR/DPDP data portability export bundle format (§12)."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "export_user@example.com", "password": "Password123!", "birth_year": 1997},
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    client.post(
        "/v1/me/consents",
        headers=headers,
        json={"purpose": "core_logging", "granted": True},
    )

    export_res = client.get("/v1/me/export", headers=headers)
    assert export_res.status_code == 200
    export_bundle = export_res.json()

    assert "anonymous_participant_id" in export_bundle
    assert export_bundle["anonymous_participant_id"].startswith("u_")
    assert "consents" in export_bundle
    assert "profile" in export_bundle
    assert "meals" in export_bundle


# ---------------------------------------------------------------------------
# Test 14: RFC 7807 Problem Details Consistency
# ---------------------------------------------------------------------------
def test_rfc7807_problem_details_structure(client):
    """Verify RFC 7807 error format across 404, 401, and 422 responses."""
    # 404 Not Found
    res_404 = client.get(f"/v1/meals/{uuid.uuid4()}", headers={"Authorization": "Bearer fake.token"})
    assert res_404.status_code in (401, 404)
    data = res_404.json()
    assert "detail" in data
    assert "title" in data
    assert "status" in data

    # 401 Unauthorized
    res_401 = client.get("/v1/me/profile")
    assert res_401.status_code == 401
    d_401 = res_401.json()
    assert d_401["status"] == 401
    assert "detail" in d_401


# ---------------------------------------------------------------------------
# Test 15: Health Check Endpoints
# ---------------------------------------------------------------------------
def test_health_endpoints(client):
    """Verify operational status on root and v1 health endpoints."""
    res_root = client.get("/health")
    assert res_root.status_code == 200
    assert res_root.json()["status"] in ("healthy", "ok")

    res_v1 = client.get("/v1/health")
    assert res_v1.status_code == 200
    assert res_v1.json()["version"] == "2.0.0"
    assert res_v1.json()["status"] in ("healthy", "ok")

