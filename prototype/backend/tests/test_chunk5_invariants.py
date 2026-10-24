"""Comprehensive Automated Verification Test Suite for Invariants I1 through I10.

Adheres strictly to MESSMATE_SPEC.md §2, §6, §11, §12, §13:
- Invariant I1: The LLM never produces nutrient numbers. Nutrients come from database lookup only.
- Invariant I2: Every estimate is a distribution, never a point (mean, sd, p10, p90 kept).
- Invariant I3: Max 1 question per meal. Max budget_total questions per day (default 3).
- Invariant I4: All questions are tap-answerable, max 4 options, no free text.
- Invariant I5: corrections is append-only. SQL UPDATE and DELETE raise errors.
- Invariant I6: Consent is per-purpose and unbundled. Independent toggles.
- Invariant I7: No account may be created or updated by a user under 18.
- Invariant I8: Never display calorie target below BMR. Never render shaming copy.
- Invariant I9: The app must work with zero questions answered.
- Invariant I10: Prompt text and model name are versioned and stored with every inference.
"""

from datetime import date, datetime, timedelta, timezone
import math
from pathlib import Path
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.safety import (
    DISCLAIMER_TEXT,
    calculate_bmr,
    clamp_goal_target,
    validate_non_shaming_text,
)
from app.core.security import create_access_token
from app.main import app
from app.models.base import Base
from app.models.food import Food
from app.models.meal import Meal, MealItem, ModelRun
from app.models.prior import UnitPrior
from app.models.question import Correction, DailyBudget, Question
from app.models.triggers import hard_delete_session, register_i5_triggers
from app.models.user import Consent, Profile, User
from app.services.allocator.engine import (
    build_tap_question,
    evaluate_budget_pressure_ask,
    get_budget_remaining,
    reserve_question_slot,
)
from app.services.resolution.engine import ResolutionService
from app.services.vlm.client import VLMClient, VLMItem, VLMOutput
from app.tasks.meal_pipeline import analyze_meal_task
from scripts.ingest.seed_nutrition_data import seed_foods


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def test_db():
    """Create an isolated, in-memory SQLite database session with I5 triggers and seeded foods."""
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
def client(test_db):
    """FastAPI TestClient with overridden get_db dependency."""
    def _override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_user(test_db):
    """Create a verified adult user and return (User, auth_headers)."""
    user = User(
        id=uuid.uuid4(),
        email="verified_adult@messmate.org",
        hashed_password="mock_argon2_hash",
        is_active=True,
    )
    test_db.add(user)
    test_db.flush()

    profile = Profile(
        user_id=user.id,
        birth_year=1995,
        sex="female",
        height_cm=165.0,
        weight_kg=60.0,
    )
    test_db.add(profile)
    test_db.commit()

    token = create_access_token(str(user.id))
    headers = {"Authorization": f"Bearer {token}"}
    return user, headers


# ---------------------------------------------------------------------------
# Invariant I1: Zero Nutrient Hallucination
# ---------------------------------------------------------------------------
def test_invariant_i1_zero_nutrient_hallucination(test_db, auth_user):
    """Invariant I1: The LLM never produces nutrient numbers.

    Verifies:
    1. Prompt text explicitly forbids calorie and macronutrient estimation.
    2. VLM schemas (VLMItem, VLMOutput) have zero nutrient fields.
    3. Nutrients only come from certified database lookup (Food table).
    """
    # 1. Prompt text inspection
    prompt_file = Path(__file__).resolve().parent.parent.parent / "prompts" / "extract_v1.txt"
    assert prompt_file.exists(), f"Prompt file {prompt_file} not found"
    prompt_text = prompt_file.read_text(encoding="utf-8")
    assert "NEVER estimate calories, macronutrients, or nutritional values (Invariant I1)" in prompt_text

    # VLM client prompt loader inspection
    vlm_client = VLMClient()
    loaded_prompt = vlm_client._load_prompt_template()
    assert "NEVER estimate calories, macronutrients, or nutritional values (Invariant I1)" in loaded_prompt

    # 2. VLM Pydantic output schemas inspection
    forbidden_nutrient_terms = ["calorie", "calories", "kcal", "protein", "fat", "carb", "carbs", "fibre", "fiber", "nutrient", "macros"]
    vlm_item_fields = set(VLMItem.model_fields.keys())
    vlm_output_fields = set(VLMOutput.model_fields.keys())

    for term in forbidden_nutrient_terms:
        assert term not in vlm_item_fields, f"VLMItem schema leaked forbidden nutrient field: '{term}'"
        assert term not in vlm_output_fields, f"VLMOutput schema leaked forbidden nutrient field: '{term}'"

    # 3. Certified database lookup verification
    user, _ = auth_user
    meal = Meal(
        user_id=user.id,
        captured_at=datetime.now(timezone.utc),
        local_date=date.today(),
        source="photo",
        status="pending",
        context="personal_home",
    )
    test_db.add(meal)
    test_db.commit()

    # Provide VLM output with label only (zero nutrients)
    vlm_mock = VLMOutput(
        plate_type="single_plate",
        items=[
            VLMItem(
                dish_label="dal makhani",
                ingredient_guess=["lentils"],
                vessel_archetype="small_bowl",
                unit_count=1.0,
                grams_estimate=150.0,
                confidence=0.95,
            )
        ],
    )

    analyzed_meal = analyze_meal_task(test_db, meal.id, mock_override=vlm_mock)
    assert analyzed_meal is not None
    assert len(analyzed_meal.items) == 1

    item = analyzed_meal.items[0]
    # Food must have been resolved to certified row in foods table
    assert item.food_id is not None
    food = test_db.get(Food, item.food_id)
    assert food is not None
    assert food.kcal_per_100g > 0.0

    # Item kcal must match: food.kcal_per_100g * (grams / 100g)
    expected_kcal = (food.kcal_per_100g / 100.0) * item.grams_mean
    assert math.isclose(item.kcal_mean, expected_kcal, rel_tol=1e-2)


# ---------------------------------------------------------------------------
# Invariant I2: Distributional Estimates (Never a Point)
# ---------------------------------------------------------------------------
def test_invariant_i2_distributional_estimates(test_db, auth_user):
    """Invariant I2: Every estimate is a distribution, never a point.

    Verifies:
    1. Meal record keeps mean, sd, and p10/p90 percentiles.
    2. MealItem keeps grams_mean, grams_sd, kcal_mean, kcal_sd, mu_log, sigma2_log.
    3. Stored spread is strictly non-zero (variance is never discarded).
    """
    user, _ = auth_user
    meal = Meal(
        user_id=user.id,
        captured_at=datetime.now(timezone.utc),
        local_date=date.today(),
        source="photo",
        status="pending",
        context="personal_home",
    )
    test_db.add(meal)
    test_db.commit()

    analyzed = analyze_meal_task(test_db, meal.id)
    assert analyzed is not None

    # Verify Meal level moments
    assert analyzed.kcal_mean > 0.0
    assert analyzed.kcal_sd > 0.0
    assert analyzed.kcal_p10 > 0.0
    assert analyzed.kcal_p90 > 0.0
    assert analyzed.kcal_p10 < analyzed.kcal_mean < analyzed.kcal_p90

    # Verify MealItem level moments
    for it in analyzed.items:
        assert it.grams_mean > 0.0
        assert it.grams_sd > 0.0
        assert it.kcal_mean > 0.0
        assert it.kcal_sd > 0.0
        assert it.mu_log is not None and it.mu_log > 0.0
        assert it.sigma2_log is not None and it.sigma2_log > 0.0


# ---------------------------------------------------------------------------
# Invariant I3: Budget Caps (Max 1/meal, Max 3/day)
# ---------------------------------------------------------------------------
def test_invariant_i3_budget_caps(test_db, auth_user):
    """Invariant I3: Max 1 question per meal. Max budget_total questions per day.

    Verifies:
    1. Database unique constraint uq_questions_meal_id blocks second question on same meal.
    2. DailyBudget and reserve_question_slot strictly enforce <= 3 questions per day.
    """
    user, _ = auth_user
    today = date.today()

    # 1. Unique constraint: max 1 question per meal
    meal = Meal(
        user_id=user.id,
        captured_at=datetime.now(timezone.utc),
        local_date=today,
        source="photo",
        status="analyzed",
    )
    test_db.add(meal)
    test_db.commit()

    q1 = Question(
        meal_id=meal.id,
        question_type="unit_count",
        prompt_text="First question",
        options=["1", "2"],
        ask_value=30.0,
        budget_remaining=2,
        policy_name="messmate_voi",
    )
    test_db.add(q1)
    test_db.commit()

    # Second question on the same meal must fail with IntegrityError
    q2 = Question(
        meal_id=meal.id,
        question_type="unit_size",
        prompt_text="Second question",
        options=["Small", "Large"],
        ask_value=25.0,
        budget_remaining=1,
        policy_name="messmate_voi",
    )
    test_db.add(q2)
    with pytest.raises(IntegrityError):
        test_db.commit()
    test_db.rollback()

    # 2. Daily budget cap (3 per day default)
    # Reserve slot 1
    ok1, rem1 = reserve_question_slot(test_db, user.id, today, budget_total=3)
    assert ok1 is True and rem1 == 2

    # Reserve slot 2
    ok2, rem2 = reserve_question_slot(test_db, user.id, today, budget_total=3)
    assert ok2 is True and rem2 == 1

    # Reserve slot 3
    ok3, rem3 = reserve_question_slot(test_db, user.id, today, budget_total=3)
    assert ok3 is True and rem3 == 0

    # Reserve slot 4 must fail!
    ok4, rem4 = reserve_question_slot(test_db, user.id, today, budget_total=3)
    assert ok4 is False and rem4 == 0

    # Allocator must refuse to ask questions when budget is 0
    gen_q = evaluate_budget_pressure_ask(
        meal_id=uuid.uuid4(),
        items=[{"food_name": "rice", "sd_kcal": 60.0}],
        budget_remaining=0,
    )
    assert gen_q is None


# ---------------------------------------------------------------------------
# Invariant I4: Tap-Only Questions (Max 4 Options, No Free Text)
# ---------------------------------------------------------------------------
def test_invariant_i4_tap_only_four_options(client, auth_user):
    """Invariant I4: All questions are tap-answerable, max 4 options, no free text.

    Verifies:
    1. Question generator restricts options list to <= 4 choices.
    2. Answering endpoint rejects free text paragraphs with HTTP 422.
    3. Answering endpoint accepts discrete integer or tap values.
    """
    user, headers = auth_user

    # 1. Question builder option cap check
    prompt, opts = build_tap_question(
        question_type="dish_disambiguation",
        food_name="soup",
        candidate_options=["sambar", "rasam", "dal", "tarka dal", "kadhi", "gravy"],
    )
    assert len(opts) <= 4

    # 2. Upload meal and test API answering validation
    upload_res = client.post(
        "/v1/meals?sync_run=true",
        headers=headers,
        data={"context": "personal_home"},
    )
    meal_id = upload_res.json()["meal_id"]

    # Free text essay answer (> 64 characters) must be rejected with HTTP 422
    long_essay = (
        "I had approximately two and a half rotis and a big bowl of sambar with "
        "lots of ghee on top and extra chili powder"
    )
    res_essay = client.post(f"/v1/meals/{meal_id}/answer", headers=headers, json={"answer": long_essay})
    assert res_essay.status_code == 422
    assert "Invariant I4" in res_essay.json()["detail"]

    # Discrete tap option (e.g. index 1) is accepted
    res_tap = client.post(f"/v1/meals/{meal_id}/answer", headers=headers, json={"answer": 1})
    assert res_tap.status_code == 200


# ---------------------------------------------------------------------------
# Invariant I5: Append-Only Corrections
# ---------------------------------------------------------------------------
def test_invariant_i5_append_only_corrections(test_db, auth_user):
    """Invariant I5: corrections table is strictly append-only.

    Verifies:
    1. Direct SQL or ORM UPDATE on corrections raises database error.
    2. Direct SQL or ORM DELETE on corrections raises database error.
    3. Hard delete session bypass allows user erasure without leaving orphaned locks.
    """
    user, _ = auth_user
    corr = Correction(
        user_id=user.id,
        field="unit_count",
        old_value={"n": 1.0},
        new_value={"n": 2.0},
        source="question",
    )
    test_db.add(corr)
    test_db.commit()
    corr_id = corr.id

    # 1. UPDATE blocked
    corr.new_value = {"n": 3.0}
    with pytest.raises(IntegrityError) as exc_update:
        test_db.commit()
    assert "Invariant I5 violated" in str(exc_update.value)
    test_db.rollback()

    # 2. DELETE blocked
    c_fetch = test_db.get(Correction, corr_id)
    test_db.delete(c_fetch)
    with pytest.raises(IntegrityError) as exc_del:
        test_db.commit()
    assert "Invariant I5 violated" in str(exc_del.value)
    test_db.rollback()

    # 3. Bypass allows deletion during complete account erasure
    with hard_delete_session(test_db):
        c_del = test_db.get(Correction, corr_id)
        test_db.delete(c_del)
        test_db.commit()

    assert test_db.get(Correction, corr_id) is None


# ---------------------------------------------------------------------------
# Invariant I6: Unbundled Consent
# ---------------------------------------------------------------------------
def test_invariant_i6_unbundled_consent(client, auth_user, test_db):
    """Invariant I6: Consent is per-purpose and unbundled.

    Verifies:
    1. Independent consent records exist for separate purposes.
    2. Revoking one consent does not mutate other consented purposes.
    """
    user, headers = auth_user

    # Grant core_logging
    c1 = client.post(
        "/v1/me/consents",
        headers=headers,
        json={"purpose": "core_logging", "granted": True, "notice_version": "v2.0"},
    )
    assert c1.status_code == 201

    # Revoke research_dataset
    c2 = client.post(
        "/v1/me/consents",
        headers=headers,
        json={"purpose": "research_dataset", "granted": False, "notice_version": "v2.0"},
    )
    assert c2.status_code == 201

    # Grant model_improvement
    c3 = client.post(
        "/v1/me/consents",
        headers=headers,
        json={"purpose": "model_improvement", "granted": True, "notice_version": "v2.0"},
    )
    assert c3.status_code == 201

    # Verify rows in database
    consents = test_db.query(Consent).filter(Consent.user_id == user.id).all()
    purposes = {c.purpose: c.granted for c in consents}

    assert purposes["core_logging"] is True
    assert purposes["research_dataset"] is False
    assert purposes["model_improvement"] is True


# ---------------------------------------------------------------------------
# Invariant I7: Minor Blocking (No User Under 18)
# ---------------------------------------------------------------------------
def test_invariant_i7_minor_blocking(client):
    """Invariant I7: No account may be created by a user under 18.

    Verifies:
    1. Age < 18 blocked at registration with HTTP 422.
    2. Age < 18 blocked at profile update with HTTP 422.
    3. Age >= 18 accepted.
    """
    curr_year = datetime.now(timezone.utc).year

    # Minor registration attempt (16 years old)
    minor_res = client.post(
        "/v1/auth/register",
        json={"email": "minor@example.com", "password": "Password123!", "birth_year": curr_year - 16},
    )
    assert minor_res.status_code == 422
    assert "Invariant I7" in minor_res.json()["detail"]

    # Boundary test (17 years old)
    seventeen_res = client.post(
        "/v1/auth/register",
        json={"email": "seventeen@example.com", "password": "Password123!", "birth_year": curr_year - 17},
    )
    assert seventeen_res.status_code == 422

    # Adult (18 years old)
    adult_res = client.post(
        "/v1/auth/register",
        json={"email": "adult18@example.com", "password": "Password123!", "birth_year": curr_year - 18},
    )
    assert adult_res.status_code == 201
    token = adult_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Minor profile update attempt
    minor_update = client.put(
        "/v1/me/profile",
        headers=headers,
        json={"birth_year": curr_year - 14},
    )
    assert minor_update.status_code == 422
    assert "Invariant I7" in minor_update.json()["detail"]


# ---------------------------------------------------------------------------
# Invariant I8: BMR Clamp and Non-Shaming Messaging
# ---------------------------------------------------------------------------
def test_invariant_i8_bmr_clamp_and_non_shaming(client, auth_user):
    """Invariant I8: Calorie targets clamped to BMR; copy sanitized of shaming phrases.

    Verifies:
    1. Mifflin-St Jeor BMR calculation accuracy.
    2. Calorie targets below BMR are clamped to BMR.
    3. Shaming phrases are caught and rejected by copy validator.
    """
    user, headers = auth_user

    # 1. BMR calculation
    bmr = calculate_bmr(birth_year=1995, sex="female", weight_kg=60.0, height_cm=165.0, current_year=2026)
    # 10*60 + 6.25*165 - 5*31 - 161 = 600 + 1031.25 - 155 - 161 = 1315.25 -> 1315.2
    assert math.isclose(bmr, 1315.2, rel_tol=1e-2)

    # 2. Clamping logic
    clamped_low = clamp_goal_target(target_kcal=900.0, bmr=bmr)
    assert clamped_low == bmr

    clamped_ok = clamp_goal_target(target_kcal=1800.0, bmr=bmr)
    assert clamped_ok == 1800.0

    # Profile update clamping via API
    res = client.put(
        "/v1/me/profile",
        headers=headers,
        json={"target_kcal": 800.0, "weight_kg": 60.0, "height_cm": 165.0},
    )
    assert res.status_code == 200
    prof = res.json()
    assert prof["clamped_target_kcal"] == prof["bmr"]
    assert prof["disclaimer"] == DISCLAIMER_TEXT

    # 3. Non-shaming copy validator
    shaming_examples = [
        "You are over budget for today!",
        "You failed your diet goal.",
        "Streak lost! You ruined your streak.",
        "Cheat day caught! Bad choice.",
        "Feel guilt about what you ate.",
        "We will punish your broken streak.",
    ]
    for phrase in shaming_examples:
        assert not validate_non_shaming_text(phrase), f"Shaming copy was not flagged: '{phrase}'"

    supportive_phrase = "Here is an observation of your intake this week. Keep up the consistent logging!"
    assert validate_non_shaming_text(supportive_phrase)


# ---------------------------------------------------------------------------
# Invariant I9: Zero Questions Answered Usable
# ---------------------------------------------------------------------------
def test_invariant_i9_zero_questions_answered_usable(client, auth_user):
    """Invariant I9: The app must work with zero questions answered.

    Verifies:
    1. A meal uploaded without answering questions is fully analyzed and available.
    2. Daily rollups aggregate meals with zero questions answered.
    """
    user, headers = auth_user

    # 1. Upload meal with sync_run=true
    upload_res = client.post(
        "/v1/meals?sync_run=true",
        headers=headers,
        data={"context": "personal_home", "meal_slot": "lunch"},
    )
    assert upload_res.status_code == 202
    meal_id = upload_res.json()["meal_id"]

    # Fetch meal directly without answering question
    detail_res = client.get(f"/v1/meals/{meal_id}", headers=headers)
    assert detail_res.status_code == 200
    detail = detail_res.json()

    assert detail["status"] == "analyzed"
    assert detail["total"]["kcal_mean"] > 0.0
    assert len(detail["items"]) > 0

    # 2. Daily rollup reflects meal without question response
    today_str = date.today().isoformat()
    day_res = client.get(f"/v1/days/{today_str}", headers=headers)
    assert day_res.status_code == 200
    day_data = day_res.json()
    assert day_data["total_kcal_mean"] >= detail["total"]["kcal_mean"]


# ---------------------------------------------------------------------------
# Invariant I10: Versioned Model Provenance
# ---------------------------------------------------------------------------
def test_invariant_i10_versioned_model_provenance(test_db, auth_user):
    """Invariant I10: Prompt text and model name are versioned and stored with every inference.

    Verifies:
    1. ModelRun record is created for every analyzed meal.
    2. ModelRun persists model_name, prompt_version, k_samples, and raw_responses.
    """
    user, _ = auth_user
    meal = Meal(
        user_id=user.id,
        captured_at=datetime.now(timezone.utc),
        local_date=date.today(),
        source="photo",
        status="pending",
        context="personal_home",
    )
    test_db.add(meal)
    test_db.commit()

    analyzed = analyze_meal_task(test_db, meal.id)
    assert analyzed is not None

    model_runs = test_db.query(ModelRun).filter(ModelRun.meal_id == meal.id).all()
    assert len(model_runs) >= 1

    run = model_runs[0]
    assert run.model_name in ("gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite")
    assert run.prompt_version == "extract_v1"
    assert run.k_samples >= 1
    assert run.raw_responses is not None
    assert "candidates" in run.raw_responses or "items" in run.raw_responses
