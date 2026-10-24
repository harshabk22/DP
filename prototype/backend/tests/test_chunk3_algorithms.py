"""Comprehensive Test Suite for Chunk 3: Algorithm Engine.

Validates:
1. Density factoring: salad (rho = 0.30) vs dal (rho = 1.05) scaling (§6.2).
2. Discretization noise decay: tau^2_eff decreasing from n=1 to n=4 (§6.2).
3. Gating & directed 3-strike outlier reset (§6.2).
4. Context gate: restaurant meals (E=0) leave priors unchanged (§6.1.2 & §6.2).
5. Variance floor: prior variance cannot drop below 0.01 (§6.2).
6. Thali-robust VoI: 5-dish meal with 50 kcal item noise qualifies for asking (§6.3).
7. Budget pressure: abundant budget asks v >= 25; tight budget applies quantile threshold (§6.3).
8. Vectorized Monte Carlo total: S = 2000 draws match analytical expectations within 1% (§6.1.3).
9. Split-conformal calibration interval bounds and non-conformity factor (§13.3).
10. Invariant I3 atomic budget reservation & DailyBudget tracking.
11. Invariant I4 option limit (<= 4 discrete tap options).
12. Sub-serving count validation guard (n < 0.5 or None).
"""

from datetime import date, datetime, timezone
import math
import uuid
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models.base import Base
from app.models.user import User
from app.services.allocator.engine import (
    V_FLOOR,
    CandidateLabel,
    GeneratedQuestion,
    ItemCandidate,
    compute_goodman_item_uncertainty,
    compute_goodman_variance,
    compute_label_mixture_moments,
    evaluate_budget_pressure_ask,
    get_budget_remaining,
    reserve_question_slot,
)
from app.services.nutrition.monte_carlo import (
    ConformalInterval,
    MealItemNutrientInput,
    MealTotalResult,
    apply_conformal_calibration,
    compute_analytical_expectations,
    compute_conformal_q_hat,
    compute_monte_carlo_total,
)
from app.services.priors.updater import (
    GLOBAL_FEAST_SHIELD_FLAG,
    INFLATED_SIGMA2,
    SIGMA2_MIN,
    PortionEstimate,
    PriorUpdateResult,
    ThaliDishInput,
    ThaliShieldResult,
    calculate_portion_distribution,
    compute_conformal_bounds,
    compute_dual_volume_estimators,
    evaluate_thali_joint_anomaly,
    update_thali_priors,
    update_unit_prior,
)


@pytest.fixture
def db_session():
    """Create an isolated in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Test 1: Density Factoring
# ---------------------------------------------------------------------------
def test_density_factoring_salad_vs_dal():
    """Verify physical density factoring decouples volume and density (MESSMATE_SPEC §6.2).

    ln(mass) = theta_{u,m} + ln(rho_d) + ln(n).
    Ratio of grams_mean for dal (rho=1.05) vs salad (rho=0.30) must be exactly 1.05 / 0.30 = 3.5.
    """
    # Standard bowl volume: 350 ml
    theta_bowl = math.log(350.0)
    sigma2_bowl = 0.04

    salad_est = calculate_portion_distribution(
        theta_um=theta_bowl,
        sigma2_um=sigma2_bowl,
        rho_d=0.30,
        n=1.0,
        kcal_per_100g=25.0,
        is_vessel=True,
        meal_noise=0.02,
    )

    dal_est = calculate_portion_distribution(
        theta_um=theta_bowl,
        sigma2_um=sigma2_bowl,
        rho_d=1.05,
        n=1.0,
        kcal_per_100g=110.0,
        is_vessel=True,
        meal_noise=0.02,
    )

    # 1. Scaling ratio of expected mass in grams must be exactly rho_dal / rho_salad = 3.5
    expected_ratio = 1.05 / 0.30  # 3.5
    actual_ratio = dal_est.grams_mean / salad_est.grams_mean
    assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-5), (
        f"Expected mass scaling ratio {expected_ratio}, got {actual_ratio}"
    )

    # 2. Difference in log-space location parameter must be ln(1.05) - ln(0.30) = ln(3.5)
    expected_log_diff = math.log(1.05) - math.log(0.30)
    actual_log_diff = dal_est.mu_log - salad_est.mu_log
    assert math.isclose(actual_log_diff, expected_log_diff, rel_tol=1e-5)

    # 3. Mass units (e.g. piece, somatic palm) do not factor food bulk density
    piece_est = calculate_portion_distribution(
        theta_um=math.log(60.0),
        sigma2_um=0.04,
        rho_d=0.30,
        n=1.0,
        kcal_per_100g=200.0,
        is_vessel=False,
        meal_noise=0.02,
    )
    assert math.isclose(piece_est.mu_log, math.log(60.0), rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Test 2: Discretization Noise Decay
# ---------------------------------------------------------------------------
def test_discretization_noise_decay():
    """Verify discretization noise tau^2_eff decreases monotonically from n=1 to n=4 (§6.2).

    Formula: tau^2_eff = tau^2_base + 1 / (12 * n^2).
    """
    prior_mu = 5.5
    prior_s2 = 0.05
    rho = 1.0

    tau_values = []
    for n in [1.0, 2.0, 3.0, 4.0]:
        # g_vlm proportional to n to keep x constant
        g_vlm = math.exp(prior_mu) * n
        res = update_unit_prior(
            prior_mu=prior_mu,
            prior_s2=prior_s2,
            g_vlm=g_vlm,
            n_user=n,
            rho_dish=rho,
            is_home=True,
            conf=1.0,
        )
        assert res.tau2_eff is not None
        tau_values.append(res.tau2_eff)

    # Strictly decreasing from n=1 to n=4
    assert tau_values[0] > tau_values[1] > tau_values[2] > tau_values[3]

    # Analytical verification:
    # tau^2_base = max(0.04, (0.20/1.0)^2) = 0.04
    # tau^2_eff(1) = 0.04 + 1/12 = 0.123333...
    # tau^2_eff(4) = 0.04 + 1/192 = 0.045208...
    # Delta = 1/12 - 1/192 = 15/192 = 0.078125
    expected_delta = (1.0 / 12.0) - (1.0 / 192.0)
    actual_delta = tau_values[0] - tau_values[3]
    assert math.isclose(actual_delta, expected_delta, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Test 3: Gating & Directed 3-Strike Outlier Reset
# ---------------------------------------------------------------------------
def test_gating_and_3strike_outlier_reset():
    """Verify dynamic 2.8-sigma gating and directed 3-strike variance reset (§6.2).

    - 3 alternating noisy meals (+3sigma, -3sigma, +3sigma) must NOT reset.
    - 3 consecutive +3sigma meals in the same direction DO reset prior variance to 0.15.
    """
    prior_mu = 5.0
    prior_s2 = 0.04
    rho = 1.0
    n = 1.0

    # tau2_base = 0.04, tau2_eff = 0.04 + 1/12 = 0.12333
    # sigma_pred = sqrt(0.04 + 0.12333) = 0.4041
    # 2.8 * sigma_pred = 1.1316
    tau2_eff = 0.04 + (1.0 / 12.0)
    sigma_pred = math.sqrt(prior_s2 + tau2_eff)
    step = 3.2 * sigma_pred  # Safely above 2.8 * sigma_pred

    # Case A: Alternating noisy meals (+, -, +)
    # Meal 1: +3.2 sigma
    g_plus = math.exp(prior_mu + step)
    res_alt1 = update_unit_prior(
        prior_mu, prior_s2, g_plus, n, rho, is_home=True,
        consecutive_rejects=0, last_reject_sign=0,
    )
    assert not res_alt1.was_updated
    assert res_alt1.reason == "gated_outlier"
    assert res_alt1.consecutive_rejects == 1
    assert res_alt1.last_reject_sign == 1
    assert res_alt1.sigma2_new == prior_s2

    # Meal 2: -3.2 sigma (direction flipped!)
    g_minus = math.exp(prior_mu - step)
    res_alt2 = update_unit_prior(
        prior_mu, prior_s2, g_minus, n, rho, is_home=True,
        consecutive_rejects=res_alt1.consecutive_rejects,
        last_reject_sign=res_alt1.last_reject_sign,
    )
    assert not res_alt2.was_updated
    assert res_alt2.consecutive_rejects == 1  # Reset to 1 because sign changed from +1 to -1
    assert res_alt2.last_reject_sign == -1
    assert res_alt2.sigma2_new == prior_s2

    # Meal 3: +3.2 sigma (direction flipped again!)
    res_alt3 = update_unit_prior(
        prior_mu, prior_s2, g_plus, n, rho, is_home=True,
        consecutive_rejects=res_alt2.consecutive_rejects,
        last_reject_sign=res_alt2.last_reject_sign,
    )
    assert not res_alt3.was_updated
    assert res_alt3.consecutive_rejects == 1
    assert res_alt3.last_reject_sign == 1
    assert res_alt3.sigma2_new == prior_s2  # Still 0.04, NOT reset!

    # Case B: 3 consecutive meals in same direction (+, +, +)
    # Strike 1
    s1 = update_unit_prior(
        prior_mu, prior_s2, g_plus, n, rho, is_home=True,
        consecutive_rejects=0, last_reject_sign=0,
    )
    assert s1.consecutive_rejects == 1
    assert s1.last_reject_sign == 1
    assert s1.sigma2_new == prior_s2

    # Strike 2
    s2 = update_unit_prior(
        prior_mu, prior_s2, g_plus, n, rho, is_home=True,
        consecutive_rejects=s1.consecutive_rejects,
        last_reject_sign=s1.last_reject_sign,
    )
    assert s2.consecutive_rejects == 2
    assert s2.last_reject_sign == 1
    assert s2.sigma2_new == prior_s2

    # Strike 3 -> Triggers 3-strike outlier reset with immediate state absorption (§3.5)
    s3 = update_unit_prior(
        prior_mu, prior_s2, g_plus, n, rho, is_home=True,
        consecutive_rejects=s2.consecutive_rejects,
        last_reject_sign=s2.last_reject_sign,
    )
    assert s3.was_updated is True
    assert s3.reason == "outlier_reset_inflated"
    # Immediate absorption: inflated prior 0.15 is absorbed into posterior using K_reset
    k_reset = 0.15 / (0.15 + s3.tau2_eff)
    expected_sigma2 = (1.0 - k_reset) * 0.15
    expected_mu = prior_mu + k_reset * (s3.x_obs_star - prior_mu)
    assert math.isclose(s3.sigma2_new, expected_sigma2, rel_tol=1e-5)
    assert math.isclose(s3.mu_new, expected_mu, rel_tol=1e-5)
    assert s3.consecutive_rejects == 0
    assert s3.last_reject_sign == 0


# ---------------------------------------------------------------------------
# Test 4: Context Gate (Restaurant Dining Frozen)
# ---------------------------------------------------------------------------
def test_context_gate_restaurant_dining_frozen():
    """Verify that commercial / away meals (E=0) never contaminate personal tableware prior (§6.1.2)."""
    prior_mu = 5.2
    prior_s2 = 0.03

    # Restaurant meal with is_home = False
    result = update_unit_prior(
        prior_mu=prior_mu,
        prior_s2=prior_s2,
        g_vlm=450.0,
        n_user=2.0,
        rho_dish=1.05,
        is_home=False,
    )

    assert result.was_updated is False
    assert result.reason == "away_dining_frozen"
    assert result.mu_new == prior_mu
    assert result.sigma2_new == prior_s2


# ---------------------------------------------------------------------------
# Test 5: Variance Floor
# ---------------------------------------------------------------------------
def test_variance_floor_enforcement():
    """Verify that prior variance sigma^2 cannot drop below 0.01 floor (§6.2)."""
    prior_mu = 5.0
    prior_s2 = 0.02
    n = 4.0
    g_vlm = math.exp(prior_mu) * n

    # Repeatedly perform 50 highly confident updates
    curr_mu = prior_mu
    curr_s2 = prior_s2

    for _ in range(50):
        res = update_unit_prior(
            prior_mu=curr_mu,
            prior_s2=curr_s2,
            g_vlm=g_vlm,
            n_user=n,
            rho_dish=1.0,
            is_home=True,
            conf=1.0,
        )
        assert res.sigma2_new >= 0.01
        curr_mu = res.mu_new
        curr_s2 = res.sigma2_new

    # Must hit and remain clamped at the 0.01 floor
    assert math.isclose(curr_s2, 0.01, rel_tol=1e-5)

    # Even if an incoming prior is smaller than 0.01, output variance is floored
    res_sub = update_unit_prior(
        prior_mu=5.0,
        prior_s2=0.005,
        g_vlm=math.exp(5.0),
        n_user=1.0,
        rho_dish=1.0,
        is_home=True,
    )
    assert res_sub.sigma2_new >= 0.01
    assert math.isclose(res_sub.sigma2_new, 0.01, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Test 6: Thali-Robust VoI (No Pythagorean Suppression)
# ---------------------------------------------------------------------------
def test_thali_robust_voi_no_pythagorean_suppression():
    """Verify 5-dish meal with 50 kcal item uncertainty qualifies for asking (§6.3).

    Classical VoI: delta_sigma = sqrt(5*50^2) - sqrt(4*50^2) = 11.8 kcal < 25 kcal (suppressed).
    Thali-robust VoI: v_i = sd_kcal = 50.0 kcal >= 25 kcal (QUALIFIES).
    """
    meal_id = uuid.uuid4()
    # 5 dishes each with 50 kcal uncertainty
    items = [
        ItemCandidate(
            meal_item_id=uuid.uuid4(),
            food_name=f"Curry_{i}",
            sd_kcal=50.0,
            unit="small_bowl",
            unit_count=1.0,
            askable=True,
        )
        for i in range(5)
    ]

    # Abundant budget B=3, meals_remaining=3 -> Pressure = 1.0
    question = evaluate_budget_pressure_ask(
        meal_id=meal_id,
        items=items,
        budget_remaining=3,
        meals_remaining=3,
    )

    assert question is not None, "Thali-robust VoI should not be suppressed on 5-dish meal!"
    assert question.ask_value == 50.0
    assert question.threshold_used == 25.0
    assert len(question.options) <= 4, "Must satisfy Invariant I4 (<= 4 options)"


# ---------------------------------------------------------------------------
# Test 7: Budget-Pressure Policy
# ---------------------------------------------------------------------------
def test_budget_pressure_policy():
    """Verify VoI threshold adaptivity under abundant vs tight budget pressure (§6.3)."""
    meal_id = uuid.uuid4()
    candidate_low = ItemCandidate(food_name="Salad", sd_kcal=20.0, askable=True)
    candidate_med = ItemCandidate(food_name="Dal", sd_kcal=40.0, askable=True)
    candidate_high = ItemCandidate(food_name="Biryani", sd_kcal=80.0, askable=True)

    # 1. Abundant budget: B=3, meals_remaining=3 -> Pressure P = 1.0
    # Candidate below floor (20.0 < 25.0) -> None
    q_below_floor = evaluate_budget_pressure_ask(
        meal_id, [candidate_low], budget_remaining=3, meals_remaining=3,
    )
    assert q_below_floor is None

    # Candidate above floor (40.0 >= 25.0) -> Asked immediately
    q_abundant = evaluate_budget_pressure_ask(
        meal_id, [candidate_med], budget_remaining=3, meals_remaining=3,
    )
    assert q_abundant is not None
    assert q_abundant.ask_value == 40.0
    assert q_abundant.threshold_used == 25.0

    # 2. Tight budget: B=1, meals_remaining=3 -> Pressure P = 1/3 = 0.333
    # Historical ask distribution: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    v_hist = [10.0 * i for i in range(1, 11)]
    # Quantile level: 1.0 - P = 1.0 - 0.333 = 0.667
    # 66.7th percentile in v_hist is ~70 kcal

    # candidate_med (40 kcal) < threshold (~70 kcal) -> Suppressed!
    q_tight_suppressed = evaluate_budget_pressure_ask(
        meal_id, [candidate_med], budget_remaining=1, meals_remaining=3, v_history=v_hist,
    )
    assert q_tight_suppressed is None

    # candidate_high (80 kcal) >= threshold (~70 kcal) -> Asked!
    q_tight_asked = evaluate_budget_pressure_ask(
        meal_id, [candidate_high], budget_remaining=1, meals_remaining=3, v_history=v_hist,
    )
    assert q_tight_asked is not None
    assert q_tight_asked.ask_value == 80.0
    assert q_tight_asked.threshold_used > 25.0

    # 3. Budget exhausted: B=0 -> Always None regardless of ask value
    q_exhausted = evaluate_budget_pressure_ask(
        meal_id, [candidate_high], budget_remaining=0, meals_remaining=3,
    )
    assert q_exhausted is None


# ---------------------------------------------------------------------------
# Test 8: Vectorized Monte Carlo Total
# ---------------------------------------------------------------------------
def test_monte_carlo_total_matches_analytical_within_1_percent():
    """Verify S = 2000 draws match exact analytical expectations within 1% (§6.1.3)."""
    items = [
        MealItemNutrientInput(
            food_name="Steamed Rice",
            mu_log=5.1,
            sigma2_log=0.03,
            kcal_per_100g=130.0,
            protein_g=2.7,
            fat_g=0.3,
            carb_g=28.0,
            fibre_g=0.4,
        ),
        MealItemNutrientInput(
            food_name="Tarka Dal",
            mu_log=5.3,
            sigma2_log=0.04,
            kcal_per_100g=105.0,
            protein_g=6.5,
            fat_g=3.8,
            carb_g=12.0,
            fibre_g=3.5,
        ),
        MealItemNutrientInput(
            food_name="Paneer Tikka",
            mu_log=4.4,
            sigma2_log=0.05,
            kcal_per_100g=265.0,
            protein_g=18.0,
            fat_g=20.0,
            carb_g=4.0,
            fibre_g=1.2,
        ),
    ]

    # Analytical closed form
    analytical = compute_analytical_expectations(items)
    true_mean = analytical["kcal_mean"]

    # Monte Carlo simulation S = 2000
    mc_result = compute_monte_carlo_total(items, num_samples=2000, seed=42)

    # 1. Error on expected calories must be strictly within 1.0%
    rel_error = abs(mc_result.kcal_mean - true_mean) / true_mean
    assert rel_error < 0.01, (
        f"Monte Carlo mean {mc_result.kcal_mean:.2f} diverged from analytical {true_mean:.2f} by {rel_error * 100:.2f}% (limit: 1.0%)"
    )

    # 2. Credible bounds ordered correctly
    assert mc_result.kcal_p10 < mc_result.kcal_mean < mc_result.kcal_p90

    # 3. Macronutrients computed
    assert mc_result.protein_mean > 0.0
    assert mc_result.fat_mean > 0.0
    assert mc_result.carb_mean > 0.0
    assert mc_result.fibre_mean > 0.0


# ---------------------------------------------------------------------------
# Test 9: Split-Conformal Calibration
# ---------------------------------------------------------------------------
def test_split_conformal_calibration():
    """Verify split-conformal calibration interval construction (§13.3)."""
    items = [
        MealItemNutrientInput(
            food_name="Dal Makhani",
            mu_log=5.2,
            sigma2_log=0.04,
            kcal_per_100g=140.0,
        )
    ]
    mc_total = compute_monte_carlo_total(items, num_samples=2000, seed=123)

    # Calibration split residuals R_i = |Y_true - Y_hat| / sigma_hat
    sample_residuals = [0.4, 0.7, 0.9, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.1]
    q_hat = compute_conformal_q_hat(sample_residuals, alpha=0.10)
    assert q_hat > 0.0

    conf_interval = apply_conformal_calibration(mc_total, q_hat=q_hat)

    # Properties
    assert conf_interval.lower >= 0.0
    assert conf_interval.lower < conf_interval.y_hat < conf_interval.upper
    assert conf_interval.q_hat == q_hat
    assert conf_interval.coverage_level == 0.90
    assert conf_interval.interval == (conf_interval.lower, conf_interval.upper)


# ---------------------------------------------------------------------------
# Test 10: Invariant I3 Atomic Lock & Budget Reservation
# ---------------------------------------------------------------------------
def test_invariant_i3_reserve_question_slot_db(db_session: Session):
    """Verify DailyBudget row creation and atomic reservation (Invariant I3)."""
    test_user = User(
        id=uuid.uuid4(),
        email="student@campus.edu",
        hashed_password="dummy_hashed_password",
        study_arm="A_fixed",
    )
    db_session.add(test_user)
    db_session.flush()

    today = date.today()

    # Initial check: 3 remaining
    rem0 = get_budget_remaining(db_session, test_user.id, today, default_total=3)
    assert rem0 == 3

    # Reserve slot 1 -> succeeds, 2 left
    ok1, rem1 = reserve_question_slot(db_session, test_user.id, today, budget_total=3)
    assert ok1 is True
    assert rem1 == 2

    # Reserve slot 2 -> succeeds, 1 left
    ok2, rem2 = reserve_question_slot(db_session, test_user.id, today, budget_total=3)
    assert ok2 is True
    assert rem2 == 1

    # Reserve slot 3 -> succeeds, 0 left
    ok3, rem3 = reserve_question_slot(db_session, test_user.id, today, budget_total=3)
    assert ok3 is True
    assert rem3 == 0

    # Reserve slot 4 -> fails! Budget exhausted
    ok4, rem4 = reserve_question_slot(db_session, test_user.id, today, budget_total=3)
    assert ok4 is False
    assert rem4 == 0

    # Verify query reflects 0 remaining
    assert get_budget_remaining(db_session, test_user.id, today) == 0


# ---------------------------------------------------------------------------
# Test 11: Invariant I4 Option Limit (<= 4 options)
# ---------------------------------------------------------------------------
def test_invariant_i4_max_four_options():
    """Verify that all generated question types strictly contain <= 4 discrete tap options."""
    meal_id = uuid.uuid4()

    # Ambiguous dish with 6 candidate labels
    item_disambig = ItemCandidate(
        food_name="Unknown Curry",
        sd_kcal=60.0,
        question_type="dish_disambiguation",
        candidate_options=["Palak Paneer", "Methi Malai", "Saag", "Hariyali", "Green Curry", "Kadhai"],
    )
    q_disambig = evaluate_budget_pressure_ask(meal_id, [item_disambig], budget_remaining=3, meals_remaining=1)
    assert q_disambig is not None
    assert len(q_disambig.options) <= 4, f"Options {q_disambig.options} exceed Invariant I4 limit of 4"

    # Unit size question
    item_size = ItemCandidate(food_name="Katori", sd_kcal=50.0, question_type="unit_size", unit="bowl")
    q_size = evaluate_budget_pressure_ask(meal_id, [item_size], budget_remaining=3, meals_remaining=1)
    assert q_size is not None
    assert len(q_size.options) <= 4

    # Unit count question
    item_count = ItemCandidate(food_name="Chapati", sd_kcal=45.0, question_type="unit_count", unit="roti")
    q_count = evaluate_budget_pressure_ask(meal_id, [item_count], budget_remaining=3, meals_remaining=1)
    assert q_count is not None
    assert len(q_count.options) <= 4


# ---------------------------------------------------------------------------
# Test 12: Sub-serving Count Guard
# ---------------------------------------------------------------------------
def test_invalid_count_guard():
    """Verify that None count or sub-0.5 count is rejected without updating prior."""
    res_none = update_unit_prior(
        prior_mu=5.0,
        prior_s2=0.04,
        g_vlm=150.0,
        n_user=None,
        rho_dish=1.0,
        is_home=True,
    )
    assert res_none.was_updated is False
    assert res_none.reason == "invalid_count"

    res_sub = update_unit_prior(
        prior_mu=5.0,
        prior_s2=0.04,
        g_vlm=150.0,
        n_user=0.2,
        rho_dish=1.0,
        is_home=True,
    )
    assert res_sub.was_updated is False
    assert res_sub.reason == "invalid_count"


# ---------------------------------------------------------------------------
# Test 13: Dual-Estimator Architecture (Jensen's Inequality Resolution)
# ---------------------------------------------------------------------------
def test_dual_estimator_architecture_jensen_bias():
    """Verify Dual-Estimator Architecture resolves Jensen's Inequality bias (§3.2).

    1. Unbiased mean estimator: V_mean = exp(mu + sigma^2/2)
    2. Conformal median estimator: V_median = exp(mu)
    3. Analytical bias: E[exp(X)] / exp(E[X]) - 1 = exp(sigma^2/2) - 1 > 0
    4. Conformal equivariant bounds: C_{0.90}(V) = [exp(mu - 1.645*sigma), exp(mu + 1.645*sigma)]
    """
    mu = 5.85  # ln(350 ml) approx 5.8579
    sigma2 = 0.15  # Up to 7.79% bias under uncalibrated tableware

    v_mean, v_median = compute_dual_volume_estimators(mu, sigma2)

    # 1. Unbiased mean must strictly exceed median by Jensen factor exp(sigma^2 / 2)
    expected_mean = math.exp(mu + sigma2 / 2.0)
    expected_median = math.exp(mu)
    assert math.isclose(v_mean, expected_mean, rel_tol=1e-6)
    assert math.isclose(v_median, expected_median, rel_tol=1e-6)
    assert v_mean > v_median

    # Jensen bias factor
    jensen_ratio = v_mean / v_median
    expected_ratio = math.exp(sigma2 / 2.0)
    assert math.isclose(jensen_ratio, expected_ratio, rel_tol=1e-6)
    percentage_bias = (jensen_ratio - 1.0) * 100.0
    # exp(0.15/2) - 1 = exp(0.075) - 1 = 7.788%
    assert math.isclose(percentage_bias, 7.788, abs_tol=0.01)

    # 2. Check PortionEstimate produces both moments
    est = calculate_portion_distribution(
        theta_um=mu,
        sigma2_um=sigma2,
        rho_d=1.0,
        n=1.0,
        kcal_per_100g=100.0,
        is_vessel=True,
    )
    assert math.isclose(est.v_mean, expected_mean, rel_tol=1e-5)
    assert math.isclose(est.v_median, expected_median, rel_tol=1e-5)
    assert est.grams_mean > est.grams_median
    assert est.kcal_mean > est.kcal_median
    assert math.isclose(est.grams_mean / est.grams_median, math.exp(est.sigma2_log / 2.0), rel_tol=1e-5)

    # 3. Conformal uncertainty bounds around median
    sigma = math.sqrt(sigma2)
    lower, upper = compute_conformal_bounds(mu, sigma, q_hat=1.645)
    assert lower < v_median < upper
    assert math.isclose(lower, math.exp(mu - 1.645 * sigma), rel_tol=1e-5)
    assert math.isclose(upper, math.exp(mu + 1.645 * sigma), rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Test 14: 2nd-Order Taylor Centering Shift
# ---------------------------------------------------------------------------
def test_second_order_taylor_centering_shift():
    """Verify 2nd-order Taylor shift x_t* = ln(V_obs / n) + 1 / (24 * n^2) (§3.3)."""
    prior_mu = 5.0
    prior_s2 = 0.04
    rho = 1.0

    shifts = []
    for n in [1.0, 2.0, 3.0, 4.0]:
        v_obs = 200.0
        g_vlm = v_obs * rho
        res = update_unit_prior(
            prior_mu=prior_mu,
            prior_s2=prior_s2,
            g_vlm=g_vlm,
            n_user=n,
            rho_dish=rho,
            is_home=True,
        )
        assert res.x_obs_star is not None
        naive_x = math.log(v_obs / n)
        shift = res.x_obs_star - naive_x
        expected_shift = 1.0 / (24.0 * (n ** 2))
        assert math.isclose(shift, expected_shift, rel_tol=1e-5), (
            f"For n={n}, expected shift {expected_shift}, got {shift}"
        )
        shifts.append(shift)

    # Shifts must decay quadratically
    assert math.isclose(shifts[0], 1.0 / 24.0, rel_tol=1e-5)   # ~0.04167
    assert math.isclose(shifts[1], 1.0 / 96.0, rel_tol=1e-5)   # ~0.01042
    assert math.isclose(shifts[2], 1.0 / 216.0, rel_tol=1e-5)  # ~0.00463
    assert math.isclose(shifts[3], 1.0 / 384.0, rel_tol=1e-5)  # ~0.00260


# ---------------------------------------------------------------------------
# Test 15: Immediate Strike-3 State Absorption
# ---------------------------------------------------------------------------
def test_immediate_strike_3_absorption():
    """Verify Strike-3 immediately absorbs trigger observation x_t* into posterior (§3.5)."""
    prior_mu = 5.0
    prior_s2 = 0.03
    rho = 1.0
    n = 1.0

    # Large observation consistently +4 sigma above prior
    tau2_eff = 0.04 + (1.0 / 12.0)
    sigma_pred = math.sqrt(prior_s2 + tau2_eff)
    g_large = math.exp(prior_mu + 3.5 * sigma_pred) * rho

    # Strike 1: gated, reject count = 1
    r1 = update_unit_prior(prior_mu, prior_s2, g_large, n, rho, is_home=True,
                           consecutive_rejects=0, last_reject_sign=0)
    assert not r1.was_updated
    assert r1.consecutive_rejects == 1

    # Strike 2: gated, reject count = 2
    r2 = update_unit_prior(prior_mu, prior_s2, g_large, n, rho, is_home=True,
                           consecutive_rejects=r1.consecutive_rejects, last_reject_sign=r1.last_reject_sign)
    assert not r2.was_updated
    assert r2.consecutive_rejects == 2

    # Strike 3: Triggers outlier reset WITH immediate state absorption
    r3 = update_unit_prior(prior_mu, prior_s2, g_large, n, rho, is_home=True,
                           consecutive_rejects=r2.consecutive_rejects, last_reject_sign=r2.last_reject_sign)
    assert r3.was_updated is True
    assert r3.reason == "outlier_reset_inflated"
    assert r3.consecutive_rejects == 0
    assert r3.last_reject_sign == 0

    # Mathematical verification of absorption:
    # K_reset = 0.15 / (0.15 + tau2_eff)
    # mu_new = prior_mu + K_reset * (x_obs_star - prior_mu)
    # sigma2_new = (1 - K_reset) * 0.15
    k_reset = INFLATED_SIGMA2 / (INFLATED_SIGMA2 + r3.tau2_eff)
    expected_mu = prior_mu + k_reset * (r3.x_obs_star - prior_mu)
    expected_s2 = max((1.0 - k_reset) * INFLATED_SIGMA2, SIGMA2_MIN)

    assert math.isclose(r3.mu_new, expected_mu, rel_tol=1e-5)
    assert math.isclose(r3.sigma2_new, expected_s2, rel_tol=1e-5)
    # mu_new has moved significantly towards the observation (NOT frozen at prior_mu!)
    assert r3.mu_new > prior_mu
    assert abs(r3.mu_new - prior_mu) > 0.5


# ---------------------------------------------------------------------------
# Test 16: Thali Joint Anomaly Shield (The Feast Trap Guard)
# ---------------------------------------------------------------------------
def test_thali_joint_anomaly_shield_feast_guard():
    """Verify Thali Joint Anomaly Shield freezes updates during multi-dish feasts (§3.6).

    If mean standardized residual across K dishes > 2.8 / sqrt(K),
    activates GLOBAL_FEAST_SHIELD_ACTIVE and freezes prior updates (Delta mu = 0, Delta sigma2 = 0).
    """
    # 1. Direct evaluate_thali_joint_anomaly check
    # Single dish (K=1): threshold = 2.8 / 1 = 2.8
    res_k1_norm = evaluate_thali_joint_anomaly([2.5])
    assert not res_k1_norm.is_shield_active
    assert res_k1_norm.threshold == 2.8

    res_k1_feast = evaluate_thali_joint_anomaly([2.9])
    assert res_k1_feast.is_shield_active
    assert res_k1_feast.flag == GLOBAL_FEAST_SHIELD_FLAG

    # 4-dish thali (K=4): threshold = 2.8 / sqrt(4) = 1.4
    # During a feast or camera scale hallucination, all dishes read +1.8 sigma
    res_k4_feast = evaluate_thali_joint_anomaly([1.8, 1.7, 1.9, 1.6])
    assert res_k4_feast.threshold == 1.4
    assert res_k4_feast.is_shield_active is True
    assert res_k4_feast.flag == GLOBAL_FEAST_SHIELD_FLAG

    # Typical meal with normal dish-to-dish fluctuations (mean z < 1.4)
    res_k4_norm = evaluate_thali_joint_anomaly([0.8, -0.4, 1.1, 0.2])
    assert res_k4_norm.is_shield_active is False
    assert res_k4_norm.flag == ""

    # 2. Multi-dish batch update: update_thali_priors
    dishes_feast = [
        ThaliDishInput(
            prior_mu=5.0,
            prior_s2=0.03,
            g_vlm=450.0,  # ~3x normal portion across all vessels!
            n_user=1.0,
            rho_dish=1.0,
            is_home=True,
        )
        for _ in range(4)
    ]

    results, shield = update_thali_priors(dishes_feast)
    assert shield.is_shield_active is True
    assert shield.flag == GLOBAL_FEAST_SHIELD_FLAG

    # ALL dish priors must be frozen!
    for r in results:
        assert r.was_updated is False
        assert r.reason == GLOBAL_FEAST_SHIELD_FLAG
        assert r.mu_new == 5.0
        assert r.sigma2_new == 0.03

    # 3. Individual update_unit_prior respects global_feast_shield flag
    res_shielded = update_unit_prior(
        prior_mu=5.2,
        prior_s2=0.04,
        g_vlm=500.0,
        n_user=1.0,
        rho_dish=1.0,
        is_home=True,
        global_feast_shield=True,
    )
    assert res_shielded.was_updated is False
    assert res_shielded.reason == GLOBAL_FEAST_SHIELD_FLAG
    assert res_shielded.mu_new == 5.2
    assert res_shielded.sigma2_new == 0.04


# ---------------------------------------------------------------------------
# Test 17: Goodman's Law of Total Variance across Label Mixture
# ---------------------------------------------------------------------------
def test_goodman_law_label_mixture_variance():
    """Verify Goodman's exact product variance under dish label mixture entropy (§3.7).

    Var(X * Y) = E[X]^2 * Var(Y) + E[Y]^2 * Var(X) + Var(X) * Var(Y).
    """
    # 1. Exact mathematical check of Goodman's formula
    # E[X] = 1.5 kcal/g, Var(X) = 0.04
    # E[Y] = 200 g, Var(Y) = 400 (sd = 20 g)
    mean_x = 1.5
    var_x = 0.04
    mean_y = 200.0
    var_y = 400.0

    # Var(XY) = (1.5)^2 * 400 + (200)^2 * 0.04 + 0.04 * 400
    #         = 2.25 * 400 + 40000 * 0.04 + 16
    #         = 900 + 1600 + 16 = 2516.0
    expected_var = (mean_x ** 2) * var_y + (mean_y ** 2) * var_x + var_x * var_y
    assert expected_var == 2516.0
    actual_var = compute_goodman_variance(mean_x, var_x, mean_y, var_y)
    assert math.isclose(actual_var, expected_var, rel_tol=1e-6)

    # 2. Label mixture moments check (Eve's Law)
    # Mixture of two curries:
    # 60% Palak Paneer (140 kcal/100g = 1.4 kcal/g, sigma=0)
    # 40% Paneer Butter Masala (240 kcal/100g = 2.4 kcal/g, sigma=0)
    labels = [
        CandidateLabel(name="Palak Paneer", probability=0.60, kcal_per_100g=140.0),
        CandidateLabel(name="Paneer Butter Masala", probability=0.40, kcal_per_100g=240.0),
    ]

    e_x, var_x = compute_label_mixture_moments(labels)
    # E[X] = 0.6 * 1.4 + 0.4 * 2.4 = 0.84 + 0.96 = 1.80 kcal/g
    assert math.isclose(e_x, 1.80, rel_tol=1e-6)
    # Var(X) = 0.6 * (1.4 - 1.8)^2 + 0.4 * (2.4 - 1.8)^2
    #        = 0.6 * 0.16 + 0.4 * 0.36 = 0.096 + 0.144 = 0.24 (kcal/g)^2
    assert math.isclose(var_x, 0.24, rel_tol=1e-6)

    # 3. Combined item uncertainty via compute_goodman_item_uncertainty
    grams_mean = 200.0
    grams_sd = 25.0
    var_y = 25.0 ** 2  # 625

    # Var(kcal) = (1.8)^2 * 625 + (200)^2 * 0.24 + 0.24 * 625
    #           = 3.24 * 625 + 40000 * 0.24 + 150 = 2025 + 9600 + 150 = 11775.0
    # sd_kcal = sqrt(11775.0) = 108.51267 kcal
    expected_sd = math.sqrt(compute_goodman_variance(1.80, 0.24, 200.0, 625.0))
    actual_sd = compute_goodman_item_uncertainty(grams_mean, grams_sd, labels)
    assert math.isclose(actual_sd, expected_sd, rel_tol=1e-5)

    # 4. Integration with evaluate_budget_pressure_ask
    meal_id = uuid.uuid4()
    candidate_ambiguous = ItemCandidate(
        food_name="Curry",
        question_type="dish_disambiguation",
        grams_mean=200.0,
        grams_sd=25.0,
        candidate_labels=labels,
        askable=True,
    )

    q = evaluate_budget_pressure_ask(
        meal_id=meal_id,
        items=[candidate_ambiguous],
        budget_remaining=3,
        meals_remaining=1,
    )
    assert q is not None
    assert math.isclose(q.ask_value, actual_sd, rel_tol=1e-4)
    assert "Palak Paneer" in q.options
    assert "Paneer Butter Masala" in q.options


# ---------------------------------------------------------------------------
# Test 18: Strict 25 kcal Chutney Theorem Cutoff
# ---------------------------------------------------------------------------
def test_chutney_theorem_strict_cutoff():
    """Verify strict 25.0 kcal Chutney Theorem cutoff (V_FLOOR = 25.0) (§3.7).

    The assistant stays silent if v_best < 25.0 kcal, even under maximal budget.
    """
    meal_id = uuid.uuid4()

    # Chutney / pickle with 24.9 kcal uncertainty (< V_FLOOR = 25.0)
    chutney = ItemCandidate(
        food_name="Mint Chutney",
        sd_kcal=24.9,
        askable=True,
    )

    # Even with abundant budget B=3 and meals_remaining=1 (Pressure = 3.0),
    # the assistant stays completely silent!
    q_silent = evaluate_budget_pressure_ask(
        meal_id=meal_id,
        items=[chutney],
        budget_remaining=3,
        meals_remaining=1,
    )
    assert q_silent is None, "Chutney Theorem violated: asked question for uncertainty < 25.0 kcal!"

    # Dish with 25.1 kcal uncertainty (>= V_FLOOR = 25.0)
    dal = ItemCandidate(
        food_name="Dal Tadka",
        sd_kcal=25.1,
        askable=True,
    )
    q_asked = evaluate_budget_pressure_ask(
        meal_id=meal_id,
        items=[dal],
        budget_remaining=3,
        meals_remaining=1,
    )
    assert q_asked is not None
    assert q_asked.ask_value == 25.1
    assert q_asked.threshold_used == V_FLOOR


# ---------------------------------------------------------------------------
# Test 19: Anti-Collapse Variance Floor Guarantees Kalman Gain
# ---------------------------------------------------------------------------
def test_anti_collapse_variance_floor_guarantees_kalman_gain():
    """Verify sigma^2_min = 0.01 rigorously guarantees Kalman gain K_t >= 0.0750 (§3.4)."""
    # At floor: sigma^2 = 0.01
    # Under maximal noise: n=1, conf=1.0 -> tau^2_base = 0.04, tau^2_eff = 0.04 + 1/12 = 0.12333
    # K_t = 0.01 / (0.01 + 0.12333) = 0.01 / 0.13333 = 0.07500
    tau2_eff_max = 0.04 + (1.0 / 12.0)
    k_min = SIGMA2_MIN / (SIGMA2_MIN + tau2_eff_max)
    assert k_min >= 0.0750, f"Kalman gain guarantee violated: {k_min} < 0.0750"

    # Run update at floor
    res = update_unit_prior(
        prior_mu=5.5,
        prior_s2=SIGMA2_MIN,
        g_vlm=math.exp(5.5),
        n_user=1.0,
        rho_dish=1.0,
        is_home=True,
        conf=1.0,
    )
    assert res.was_updated is True
    # Posterior variance must remain clamped at SIGMA2_MIN
    assert math.isclose(res.sigma2_new, SIGMA2_MIN, rel_tol=1e-5)

