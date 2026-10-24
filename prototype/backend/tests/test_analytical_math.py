"""Unit tests for closed-form analytical nutrient engine (p01.md §4.1).

Validates:
1. Exact closed-form expectation and variance composition.
2. High-speed sub-millisecond execution (< 100 microseconds).
3. Concordance with large-sample Monte Carlo compositor (< 1.5% sampling gap).
4. Correct multi-item uncertainty propagation and credible interval bounds [p10, p90].
5. Edge cases: empty list, zero mass, direct manual user grams.
"""

import math
import time
import numpy as np
import pytest

from app.services.allocator.engine import (
    V_FLOOR,
    CandidateLabel,
    ItemCandidate,
    compute_goodman_item_uncertainty,
    compute_goodman_variance,
    compute_label_mixture_moments,
    evaluate_budget_pressure_ask,
)
from app.services.nutrition.monte_carlo import (
    MealItemNutrientInput,
    MealTotalResult,
    compute_analytical_expectations,
    compute_analytical_total,
    compute_monte_carlo_total,
)
from app.services.priors.updater import (
    INFLATED_SIGMA2,
    SIGMA2_MIN,
    calculate_portion_distribution,
    compute_dual_volume_estimators,
    compute_lognormal_analytical_moments,
    update_unit_prior,
)


def test_analytical_total_empty():
    """Empty item sequence must return zeroed result."""
    res = compute_analytical_total([])
    assert isinstance(res, MealTotalResult)
    assert res.kcal_mean == 0.0
    assert res.kcal_sd == 0.0
    assert res.kcal_p10 == 0.0
    assert res.kcal_p90 == 0.0
    assert res.protein_mean == 0.0
    assert res.fat_mean == 0.0
    assert res.carb_mean == 0.0
    assert res.fibre_mean == 0.0
    assert res.samples_count == 0


def test_analytical_total_exact_math():
    """Verify exact theoretical moments for lognormal and manual grams."""
    # Item 1: Lognormal parameters
    mu1 = 5.0
    s2_1 = 0.04
    kcal_per_100g_1 = 150.0
    protein_1 = 8.0
    fat_1 = 5.0
    carb_1 = 20.0
    fibre_1 = 3.0

    # Theoretical mass moments
    mean_g1 = math.exp(mu1 + s2_1 / 2.0)
    var_g1 = math.exp(2.0 * mu1 + s2_1) * (math.exp(s2_1) - 1.0)
    sd_g1 = math.sqrt(var_g1)

    # Item 2: Direct user manual grams
    g2 = 250.0
    sd_g2 = 15.0
    kcal_per_100g_2 = 120.0
    protein_2 = 3.0
    fat_2 = 1.0
    carb_2 = 25.0
    fibre_2 = 1.5

    items = [
        MealItemNutrientInput(
            food_name="Dish A",
            mu_log=mu1,
            sigma2_log=s2_1,
            kcal_per_100g=kcal_per_100g_1,
            protein_g=protein_1,
            fat_g=fat_1,
            carb_g=carb_1,
            fibre_g=fibre_1,
        ),
        MealItemNutrientInput(
            food_name="Dish B",
            grams_mean=g2,
            grams_sd=sd_g2,
            kcal_per_100g=kcal_per_100g_2,
            protein_g=protein_2,
            fat_g=fat_2,
            carb_g=carb_2,
            fibre_g=fibre_2,
        ),
    ]

    res = compute_analytical_total(items)

    # Theoretical kcal
    ratio1 = mean_g1 / 100.0
    ratio_sd1 = sd_g1 / 100.0
    ratio2 = g2 / 100.0
    ratio_sd2 = sd_g2 / 100.0

    expected_kcal = kcal_per_100g_1 * ratio1 + kcal_per_100g_2 * ratio2
    expected_kcal_var = (kcal_per_100g_1 * ratio_sd1) ** 2 + (kcal_per_100g_2 * ratio_sd2) ** 2
    expected_kcal_sd = math.sqrt(expected_kcal_var)

    assert abs(res.kcal_mean - expected_kcal) < 0.2
    assert abs(res.kcal_sd - expected_kcal_sd) < 0.2
    assert res.kcal_p10 < res.kcal_mean < res.kcal_p90
    assert abs(res.kcal_p10 - max(0.0, round(res.kcal_mean - 1.28155 * res.kcal_sd, 1))) < 0.1
    assert abs(res.kcal_p90 - round(res.kcal_mean + 1.28155 * res.kcal_sd, 1)) < 0.1

    # Theoretical protein
    expected_protein = protein_1 * ratio1 + protein_2 * ratio2
    assert abs(res.protein_mean - expected_protein) < 0.2


def test_analytical_concordance_with_monte_carlo():
    """Analytical expectation must match Monte Carlo expectation within sampling error (<1.5%)."""
    items = [
        MealItemNutrientInput(
            food_name="Rice",
            mu_log=5.4,
            sigma2_log=0.03,
            kcal_per_100g=130.0,
            protein_g=2.7,
            fat_g=0.3,
            carb_g=28.0,
            fibre_g=0.4,
        ),
        MealItemNutrientInput(
            food_name="Dal",
            mu_log=5.1,
            sigma2_log=0.05,
            kcal_per_100g=105.0,
            protein_g=7.0,
            fat_g=3.5,
            carb_g=12.0,
            fibre_g=4.0,
        ),
        MealItemNutrientInput(
            food_name="Paneer Curry",
            mu_log=4.8,
            sigma2_log=0.06,
            kcal_per_100g=210.0,
            protein_g=11.0,
            fat_g=16.0,
            carb_g=6.0,
            fibre_g=1.2,
        ),
    ]

    analytical = compute_analytical_total(items)
    mc = compute_monte_carlo_total(items, num_samples=10000, seed=42)

    rel_error_kcal = abs(analytical.kcal_mean - mc.kcal_mean) / analytical.kcal_mean
    assert rel_error_kcal < 0.015, f"Relative difference {rel_error_kcal * 100:.2f}% exceeds 1.5%"

    rel_error_protein = abs(analytical.protein_mean - mc.protein_mean) / analytical.protein_mean
    assert rel_error_protein < 0.02, f"Protein difference {rel_error_protein * 100:.2f}% exceeds 2%"


def test_analytical_performance_sub_millisecond():
    """Analytical computation for typical 5-dish meal must run in < 100 microseconds."""
    items = [
        MealItemNutrientInput(
            food_name=f"Dish {i}",
            mu_log=5.0,
            sigma2_log=0.04,
            kcal_per_100g=150.0,
            protein_g=5.0,
            fat_g=5.0,
            carb_g=20.0,
            fibre_g=2.0,
        )
        for i in range(5)
    ]

    # Warm-up
    for _ in range(10):
        compute_analytical_total(items)

    iterations = 500
    t0 = time.perf_counter()
    for _ in range(iterations):
        compute_analytical_total(items)
    t1 = time.perf_counter()

    avg_time_us = ((t1 - t0) / iterations) * 1e6
    assert avg_time_us < 100.0, f"Average latency {avg_time_us:.2f} µs exceeded 100 µs target"


# ---------------------------------------------------------------------------
# Audit Test 1: Exact Lognormal Moments & Variance Proof
# ---------------------------------------------------------------------------
def test_exact_lognormal_moments_and_variance_proof():
    """Mathematical integrity proof for log-normal portion distribution (§3.2).

    Given Y ~ Lognormal(mu, sigma^2):
    1. Unbiased Mean: E[Y] = exp(mu + sigma^2 / 2)
    2. Conformal Median: Med[Y] = exp(mu)
    3. Variance: Var(Y) = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    4. Standard deviation: SD(Y) = sqrt(Var(Y)) = E[Y] * sqrt(exp(sigma^2) - 1)
    """
    for mu in [3.0, 5.0, 5.85, 6.5]:
        for s2 in [0.01, 0.04, 0.10, 0.15]:
            mean, median, var, sd = compute_lognormal_analytical_moments(mu, s2)

            # 1. Unbiased mean identity
            expected_mean = math.exp(mu + s2 / 2.0)
            assert math.isclose(mean, expected_mean, rel_tol=1e-12)

            # 2. Conformal median identity
            expected_median = math.exp(mu)
            assert math.isclose(median, expected_median, rel_tol=1e-12)

            # 3. Exact variance identity
            expected_var = (math.exp(s2) - 1.0) * math.exp(2.0 * mu + s2)
            assert math.isclose(var, expected_var, rel_tol=1e-12)

            # 4. Standard deviation relationship
            expected_sd = expected_mean * math.sqrt(math.exp(s2) - 1.0)
            assert math.isclose(sd, expected_sd, rel_tol=1e-12)
            assert math.isclose(sd, math.sqrt(var), rel_tol=1e-12)

            # 5. Jensen's inequality bias
            assert mean > median
            jensen_gap = mean - median
            assert jensen_gap > 0.0

    # Test PortionEstimate outputs match exact analytical moments
    est = calculate_portion_distribution(
        theta_um=5.8579,
        sigma2_um=0.04,
        rho_d=1.05,
        n=2.0,
        kcal_per_100g=120.0,
        is_vessel=True,
    )
    assert math.isclose(est.grams_var, (math.exp(est.sigma2_log) - 1.0) * math.exp(2.0 * est.mu_log + est.sigma2_log), rel_tol=1e-6)
    assert math.isclose(est.grams_sd, math.sqrt(est.grams_var), rel_tol=1e-6)
    assert math.isclose(est.kcal_var, est.grams_var * (1.20 ** 2), rel_tol=1e-6)
    assert math.isclose(est.kcal_sd, math.sqrt(est.kcal_var), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Audit Test 2: Exact Goodman's Product Variance Proof
# ---------------------------------------------------------------------------
def test_exact_goodmans_product_variance_proof():
    """Empirical & analytical proof of Goodman's (1960) Product Variance (§3.7).

    For independent random variables X (kcal/g) and Y (mass in g):
    Var(XY) = E[X]^2 * Var(Y) + E[Y]^2 * Var(X) + Var(X) * Var(Y).
    """
    mean_x = 1.45  # kcal/g
    var_x = 0.05   # (kcal/g)^2
    mean_y = 220.0 # g
    var_y = 400.0  # g^2 (sd = 20g)

    # Closed-form Goodman variance
    goodman_var = compute_goodman_variance(mean_x, var_x, mean_y, var_y)
    expected_goodman = (mean_x ** 2) * var_y + (mean_y ** 2) * var_x + var_x * var_y
    assert math.isclose(goodman_var, expected_goodman, rel_tol=1e-12)

    # Monte Carlo empirical verification with 1,000,000 draws
    rng = np.random.default_rng(seed=1337)
    x_draws = rng.normal(loc=mean_x, scale=math.sqrt(var_x), size=1_000_000)
    y_draws = rng.normal(loc=mean_y, scale=math.sqrt(var_y), size=1_000_000)
    xy_draws = x_draws * y_draws

    empirical_var = float(np.var(xy_draws, ddof=1))
    # Empirical variance matches Goodman formula within 0.5% Monte Carlo sampling error
    rel_error = abs(empirical_var - goodman_var) / goodman_var
    assert rel_error < 0.005, f"Goodman formula {goodman_var} diverged from empirical {empirical_var} by {rel_error*100:.3f}%"


# ---------------------------------------------------------------------------
# Audit Test 3: 2nd-Order Taylor Shift & Discretization Decay (No Heuristics)
# ---------------------------------------------------------------------------
def test_second_order_taylor_shift_and_discretization_decay_no_heuristics():
    """Prove 1/(24*n^2) and 1/(12*n^2) derive strictly from uniform discretization (§3.3).

    For tap count n^* = n + epsilon, epsilon ~ U[-1/2, 1/2]:
    E[epsilon] = 0, Var(epsilon) = 1/12.
    E[ln(1 + epsilon/n)] approx E[epsilon/n - epsilon^2 / (2n^2)] = -1/(24n^2).
    Unbiased log-volume requires centering shift: + 1/(24n^2).
    Variance of epsilon/n is Var(epsilon) / n^2 = 1/(12n^2).
    """
    for n in [1.0, 2.0, 3.0, 4.0]:
        res = update_unit_prior(
            prior_mu=5.0,
            prior_s2=0.04,
            g_vlm=250.0,
            n_user=n,
            rho_dish=1.0,
            is_home=True,
            conf=1.0,
        )
        naive_log = math.log(250.0 / n)
        # Shift must be exactly + 1 / (24 * n^2)
        assert math.isclose(res.x_obs_star - naive_log, 1.0 / (24.0 * (n ** 2)), rel_tol=1e-10)

        # Discretization noise addition must be exactly + 1 / (12 * n^2)
        base_tau2 = 0.04  # max(0.04, (0.20/1.0)^2)
        assert math.isclose(res.tau2_eff - base_tau2, 1.0 / (12.0 * (n ** 2)), rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Audit Test 4: Anti-Collapse Floor & Strike-3 Immediate Absorption Proof
# ---------------------------------------------------------------------------
def test_anti_collapse_floor_and_strike_3_absorption_proof():
    """Prove sigma^2_min = 0.01 guarantees K_t >= 0.0750 and Strike-3 absorbs observation (§3.4, §3.5)."""
    # 1. Floor proof
    # Maximum observation noise under tap count n=1, conf=1:
    tau2_eff_max = 0.04 + (1.0 / 12.0)
    k_min = SIGMA2_MIN / (SIGMA2_MIN + tau2_eff_max)
    # Proof: 0.01 / (0.01 + 0.12333) = 0.01 / 0.13333 = 3/40 = 0.07500
    assert math.isclose(k_min, 0.07500, rel_tol=1e-4)
    assert k_min >= 0.0750

    # 2. Strike-3 Immediate State Absorption
    prior_mu = 5.0
    prior_s2 = 0.03
    g_large = 600.0  # Massive outlier

    # Step through 3 consecutive outliers
    s1 = update_unit_prior(prior_mu, prior_s2, g_large, 1.0, 1.0, is_home=True,
                           consecutive_rejects=0, last_reject_sign=0)
    s2 = update_unit_prior(prior_mu, prior_s2, g_large, 1.0, 1.0, is_home=True,
                           consecutive_rejects=s1.consecutive_rejects, last_reject_sign=s1.last_reject_sign)
    s3 = update_unit_prior(prior_mu, prior_s2, g_large, 1.0, 1.0, is_home=True,
                           consecutive_rejects=s2.consecutive_rejects, last_reject_sign=s2.last_reject_sign)

    assert s3.reason == "outlier_reset_inflated"
    # Inflated prior 0.15 is absorbed immediately
    k_reset = INFLATED_SIGMA2 / (INFLATED_SIGMA2 + s3.tau2_eff)
    expected_mu = prior_mu + k_reset * (s3.x_obs_star - prior_mu)
    expected_s2 = max((1.0 - k_reset) * INFLATED_SIGMA2, SIGMA2_MIN)

    assert math.isclose(s3.mu_new, expected_mu, rel_tol=1e-6)
    assert math.isclose(s3.sigma2_new, expected_s2, rel_tol=1e-6)
    assert s3.mu_new != prior_mu  # Never frozen at prior_mu!


# ---------------------------------------------------------------------------
# Audit Test 5: Chutney Theorem Strict 25 kcal Floor Cutoff
# ---------------------------------------------------------------------------
def test_chutney_theorem_strict_cutoff_invariance():
    """Prove assistant stays strictly silent when v_best < 25.0 kcal (§3.7).

    Guarantees total daily unasked uncertainty < 75 kcal/day (sigma < 43.3 kcal/day),
    submerging unasked uncertainty beneath physiological metabolic noise.
    """
    meal_id = pytest.importorskip("uuid").uuid4()

    # Candidate with 24.99 kcal uncertainty (below 25.0 kcal floor)
    below = ItemCandidate(food_name="Chutney", sd_kcal=24.99, askable=True)
    # Even under maximum budget pressure (P = 3.0), must return None
    res_below = evaluate_budget_pressure_ask(meal_id, [below], budget_remaining=3, meals_remaining=1)
    assert res_below is None

    # Candidate with 25.00 kcal uncertainty (at or above 25.0 kcal floor)
    at_floor = ItemCandidate(food_name="Chutney", sd_kcal=25.00, askable=True)
    res_at = evaluate_budget_pressure_ask(meal_id, [at_floor], budget_remaining=3, meals_remaining=1)
    assert res_at is not None
    assert res_at.ask_value == 25.0
    assert res_at.threshold_used == V_FLOOR

