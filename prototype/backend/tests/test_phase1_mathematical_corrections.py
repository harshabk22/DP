"""Unit and integration tests for Phase 1 mathematical & statistical corrections.

Validates:
1. Non-additive quantile summation fix in daily rollups (rollups.py):
   Variance accumulation total_sd = sqrt(sum(sd_i^2)) and conformal calibration
   versus incorrect linear quantile summation.
2. Split-conformal calibration scaling (q_hat) in question answering recalculation (questions.py).
3. Clamping 2nd-order Taylor shift and discretization variance for fractional counts n < 1.0 (updater.py).
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
from app.main import app
from app.models.base import Base
from app.models.food import Food
from app.models.meal import Meal, MealItem
from app.models.prior import UnitPrior
from app.models.question import Question
from app.models.triggers import register_i5_triggers
from app.models.user import Profile, User
from app.services.conformal.engine import (
    DEFAULT_CONFORMAL_Q_90,
    ConformalCalibrationService,
)
from app.services.priors.updater import (
    TAU2_BASE_MIN,
    ThaliDishInput,
    update_thali_priors,
    update_unit_prior,
)
from scripts.ingest.seed_nutrition_data import seed_foods


# ---------------------------------------------------------------------------
# Task 3: Taylor Shift Clamping for Fractional Counts (updater.py)
# ---------------------------------------------------------------------------

def test_taylor_shift_clamping_for_fractional_counts():
    """Verify Taylor shift is clamped to 0.0 and tau2_disc to 0.04 when n < 1.0."""
    prior_mu = math.log(300.0)  # ~5.70378
    prior_s2 = 0.04
    rho_dish = 1.0  # 1.0 g/ml
    is_home = True

    # Case 1: Fractional count n = 0.5 (e.g. half portion)
    n_fractional = 0.5
    # Let observed grams correspond to 150 ml (so v_obs = 150 ml, v_obs / n = 300 ml)
    g_vlm_frac = 150.0
    v_obs_frac = g_vlm_frac / rho_dish  # 150.0 ml

    res_frac = update_unit_prior(
        prior_mu=prior_mu,
        prior_s2=prior_s2,
        g_vlm=g_vlm_frac,
        n_user=n_fractional,
        rho_dish=rho_dish,
        is_home=is_home,
        conf=1.0,
    )

    assert res_frac.was_updated is True
    # Without clamp: x_obs_star would be ln(150 / 0.5) + 1/(24 * 0.25) = ln(300) + 0.16667
    # With clamp: x_obs_star must equal ln(150 / 0.5) + 0.0 = ln(300)
    expected_x_frac = math.log(v_obs_frac / n_fractional)  # math.log(300.0)
    assert res_frac.x_obs_star == pytest.approx(expected_x_frac, abs=1e-6)

    # For tau2_eff: tau2_base = max(0.04, (0.20 / 1.0)^2) = 0.04
    # With clamp: tau2_disc = 0.04, so tau2_eff = 0.04 + 0.04 = 0.08
    # Without clamp: tau2_disc would be 1 / (12 * 0.25) = 0.3333, tau2_eff = 0.3733
    expected_tau2_eff = 0.04 + 0.04
    assert res_frac.tau2_eff == pytest.approx(expected_tau2_eff, abs=1e-6)


def test_taylor_shift_retained_for_integer_counts():
    """Verify Taylor shift 1/(24*n^2) and discretization 1/(12*n^2) are retained for n >= 1.0."""
    prior_mu = math.log(300.0)
    prior_s2 = 0.04
    rho_dish = 1.0
    is_home = True

    n_integer = 1.0
    g_vlm_int = 300.0
    v_obs_int = g_vlm_int / rho_dish

    res_int = update_unit_prior(
        prior_mu=prior_mu,
        prior_s2=prior_s2,
        g_vlm=g_vlm_int,
        n_user=n_integer,
        rho_dish=rho_dish,
        is_home=is_home,
        conf=1.0,
    )

    assert res_int.was_updated is True
    expected_x_int = math.log(v_obs_int / n_integer) + 1.0 / (24.0 * (n_integer ** 2))
    assert res_int.x_obs_star == pytest.approx(expected_x_int, abs=1e-6)

    expected_tau2_int = 0.04 + 1.0 / (12.0 * (n_integer ** 2))
    assert res_int.tau2_eff == pytest.approx(expected_tau2_int, abs=1e-6)


def test_thali_priors_taylor_shift_fractional_clamp():
    """Verify update_thali_priors clamps Taylor shift and tau2_disc for fractional dishes."""
    dishes = [
        ThaliDishInput(
            dish_label="half_dal",
            prior_mu=math.log(200.0),
            prior_s2=0.04,
            g_vlm=100.0,
            n_user=0.5,
            rho_dish=1.0,
            is_home=True,
            conf=1.0,
        )
    ]
    results, shield = update_thali_priors(dishes)
    assert len(results) == 1
    assert results[0].was_updated is True
    expected_x = math.log(100.0 / 0.5)  # ln(200), no +1/(24*0.25) shift
    assert results[0].x_obs_star == pytest.approx(expected_x, abs=1e-6)
    assert results[0].tau2_eff == pytest.approx(0.08, abs=1e-6)


# ---------------------------------------------------------------------------
# Task 1 & 2 Integration Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_db():
    """Create isolated in-memory SQLite database session."""
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
def client_app(test_db):
    """FastAPI TestClient with overridden get_db."""
    def _override():
        yield test_db

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_auth(client_app):
    """Register and authenticate user."""
    email = f"math_tester_{datetime.now(timezone.utc).timestamp()}@example.com"
    pwd = "SecurePassword123!"
    reg_res = client_app.post(
        "/v1/auth/register",
        json={"email": email, "password": pwd, "birth_year": 1995, "sex": "male", "goal": "maintain"},
    )
    assert reg_res.status_code == 201
    token = reg_res.json()["access_token"]
    user_id = reg_res.json()["user_id"]
    return {"Authorization": f"Bearer {token}"}, user_id


# ---------------------------------------------------------------------------
# Task 1: Non-Additive Quantile Summation in Daily Rollups (rollups.py)
# ---------------------------------------------------------------------------

def test_daily_rollup_non_additive_quantile_conformal_bounds(client_app, test_auth, test_db):
    """Verify daily rollup accumulates variance (not linearly summing quantiles) and calibrates."""
    headers, user_id = test_auth
    today_date = datetime.now(timezone.utc).date()

    # Manually create two analyzed meals for today with distinct kcal_sd
    # Meal 1: kcal_mean = 500, kcal_sd = 40
    # Meal 2: kcal_mean = 700, kcal_sd = 30
    meal1 = Meal(
        id=uuid.uuid4(),
        user_id=uuid.UUID(user_id),
        captured_at=datetime.now(timezone.utc),
        local_date=today_date,
        meal_slot="lunch",
        source="text",
        status="analyzed",
        context="personal_home",
        kcal_mean=500.0,
        kcal_sd=40.0,
        kcal_p10=round(500.0 - DEFAULT_CONFORMAL_Q_90 * 40.0, 1),  # 426.3
        kcal_p90=round(500.0 + DEFAULT_CONFORMAL_Q_90 * 40.0, 1),  # 573.7
    )
    meal2 = Meal(
        id=uuid.uuid4(),
        user_id=uuid.UUID(user_id),
        captured_at=datetime.now(timezone.utc),
        local_date=today_date,
        meal_slot="dinner",
        source="text",
        status="analyzed",
        context="personal_home",
        kcal_mean=700.0,
        kcal_sd=30.0,
        kcal_p10=round(700.0 - DEFAULT_CONFORMAL_Q_90 * 30.0, 1),  # 644.7
        kcal_p90=round(700.0 + DEFAULT_CONFORMAL_Q_90 * 30.0, 1),  # 755.3
    )
    test_db.add_all([meal1, meal2])
    test_db.commit()

    # Mathematical expectation:
    # total_kcal = 1200.0
    # total_kcal_var = 40^2 + 30^2 = 1600 + 900 = 2500
    # total_sd = sqrt(2500) = 50.0
    # Calibrated daily bounds:
    # expected_p10 = 1200.0 - 1.842 * 50.0 = 1107.9
    # expected_p90 = 1200.0 + 1.842 * 50.0 = 1292.1
    # Linear sum (INCORRECT):
    # linear_p10 = 426.3 + 644.7 = 1071.0
    # linear_p90 = 573.7 + 755.3 = 1329.0
    expected_daily_bounds = ConformalCalibrationService.calibrate_meal_bounds(
        kcal_mean=1200.0,
        kcal_sd=50.0,
    )

    res = client_app.get(f"/v1/days/{today_date.isoformat()}", headers=headers)
    assert res.status_code == 200
    rollup = res.json()

    assert rollup["total_kcal_mean"] == 1200.0
    assert rollup["total_kcal_p10"] == expected_daily_bounds.kcal_p10
    assert rollup["total_kcal_p90"] == expected_daily_bounds.kcal_p90

    # Explicitly check that it differs from the erroneous linear sum
    linear_p10 = meal1.kcal_p10 + meal2.kcal_p10
    linear_p90 = meal1.kcal_p90 + meal2.kcal_p90
    assert rollup["total_kcal_p10"] != linear_p10
    assert rollup["total_kcal_p90"] != linear_p90
    # Variance accumulation produces tighter (statistically proper) intervals than linear sum:
    assert rollup["total_kcal_p10"] > linear_p10
    assert rollup["total_kcal_p90"] < linear_p90


# ---------------------------------------------------------------------------
# Task 2: Conformal Scaling Restoration in Question Answering (questions.py)
# ---------------------------------------------------------------------------

def test_question_answering_restores_conformal_calibration(client_app, test_auth, test_db):
    """Verify answering a question applies ConformalCalibrationService to meal moments."""
    headers, user_id = test_auth
    today_date = datetime.now(timezone.utc).date()
    uid = uuid.UUID(user_id)

    food = test_db.query(Food).first()

    meal = Meal(
        id=uuid.uuid4(),
        user_id=uid,
        captured_at=datetime.now(timezone.utc),
        local_date=today_date,
        meal_slot="lunch",
        source="photo",
        status="analyzed",
        context="personal_home",
        kcal_mean=400.0,
        kcal_sd=40.0,
        kcal_p10=326.3,
        kcal_p90=473.7,
    )
    test_db.add(meal)
    test_db.flush()

    item = MealItem(
        id=uuid.uuid4(),
        meal_id=meal.id,
        dish_label_raw=food.name_en,
        food_id=food.id,
        resolution_tier="exact",
        unit="bowl",
        unit_count=1.0,
        grams_mean=150.0,
        grams_sd=15.0,
        kcal_mean=200.0,
        kcal_sd=20.0,
        mu_log=math.log(150.0),
        sigma2_log=0.01,
        portion_source="prior",
    )
    test_db.add(item)
    test_db.flush()

    # Prior for user tableware
    prior = UnitPrior(
        user_id=uid,
        unit="bowl",
        context="personal_home",
        mu_log=math.log(150.0),
        sigma2_log=0.04,
        n_obs=0,
        consecutive_rejects=0,
        last_reject_sign=0,
    )
    test_db.add(prior)
    test_db.flush()

    question = Question(
        id=uuid.uuid4(),
        meal_id=meal.id,
        meal_item_id=item.id,
        question_type="unit_count",
        prompt_text="Did you have 1 bowl or 2 bowls?",
        options=[{"label": "1 bowl", "value": 1.0}, {"label": "2 bowls", "value": 2.0}],
        ask_value=35.0,
        threshold_used=20.0,
        budget_remaining=2,
        policy_name="messmate_voi",
    )
    test_db.add(question)
    test_db.commit()

    # Answer the question: confirmed 2 bowls via POST /v1/meals/{meal_id}/answer
    ans_res = client_app.post(
        f"/v1/meals/{meal.id}/answer",
        headers=headers,
        json={"answer": 2.0},
    )
    assert ans_res.status_code == 200

    # Reload meal from DB
    test_db.refresh(meal)

    # Conformal calibration guarantee:
    # meal.kcal_p90 = round(meal.kcal_mean + DEFAULT_CONFORMAL_Q_90 * meal.kcal_sd, 1)
    # meal.kcal_p10 = max(0.0, round(meal.kcal_mean - DEFAULT_CONFORMAL_Q_90 * meal.kcal_sd, 1))
    expected_bounds = ConformalCalibrationService.calibrate_meal_bounds(
        kcal_mean=meal.kcal_mean,
        kcal_sd=meal.kcal_sd,
    )
    assert meal.kcal_p10 == expected_bounds.kcal_p10
    assert meal.kcal_p90 == expected_bounds.kcal_p90

    # Ensure it is NOT the uncalibrated Gaussian nominal (Z=1.28155)
    nominal_p90 = round(meal.kcal_mean + 1.28155 * meal.kcal_sd, 1)
    nominal_p10 = max(0.0, round(meal.kcal_mean - 1.28155 * meal.kcal_sd, 1))
    assert meal.kcal_p90 > nominal_p90
    assert meal.kcal_p10 < nominal_p10
