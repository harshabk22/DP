"""Vectorized Monte Carlo Nutrient Compositor and Split-Conformal Calibration.

Adheres to MESSMATE_SPEC.md §6.1.3 and §13.3:
- S = 2000 vectorized log-normal draws per meal item.
- Closed-form and empirical moments for calories and macronutrients (protein, fat, carb, fibre).
- Empirical credible bounds [p10, p90].
- Split-conformal prediction interval calibration with finite-sample coverage guarantee:
    [p10, p90]_calibrated = [Y_hat - q_hat * sigma_hat, Y_hat + q_hat * sigma_hat]
    P(Y_true in [p10, p90]) >= 0.90
"""

import math
from typing import Any, Optional, Sequence
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class MealItemNutrientInput(BaseModel):
    """Nutrient parameters for a single meal component."""

    model_config = ConfigDict(from_attributes=True)

    food_name: str = Field(default="dish", description="Identified food name")
    mu_log: float = Field(default=0.0, description="Location parameter mu_log of log-normal gram distribution")
    sigma2_log: float = Field(default=0.0, description="Scale parameter sigma^2_log of log-normal gram distribution")
    kcal_per_100g: float = Field(default=0.0, description="Energy density per 100g")
    protein_g: Optional[float] = Field(default=0.0, description="Protein content in g per 100g")
    fat_g: Optional[float] = Field(default=0.0, description="Fat content in g per 100g")
    carb_g: Optional[float] = Field(default=0.0, description="Carbohydrate content in g per 100g")
    fibre_g: Optional[float] = Field(default=0.0, description="Dietary fibre content in g per 100g")
    grams_mean: Optional[float] = Field(default=None, description="Direct mean mass in grams")
    grams_sd: Optional[float] = Field(default=None, description="Direct standard deviation of mass in grams")


class MealTotalResult(BaseModel):
    """Aggregate multi-dish probabilistic nutrient composition result."""

    model_config = ConfigDict(from_attributes=True)

    # Calorie summary
    kcal_mean: float = Field(..., description="Empirical mean total calories")
    kcal_sd: float = Field(..., description="Calorie uncertainty (standard deviation)")
    kcal_p10: float = Field(..., description="Nominal 10th percentile")
    kcal_p90: float = Field(..., description="Nominal 90th percentile")

    # Macronutrient summaries (in grams)
    protein_mean: float = 0.0
    protein_sd: float = 0.0
    protein_p10: float = 0.0
    protein_p90: float = 0.0

    fat_mean: float = 0.0
    fat_sd: float = 0.0
    fat_p10: float = 0.0
    fat_p90: float = 0.0

    carb_mean: float = 0.0
    carb_sd: float = 0.0
    carb_p10: float = 0.0
    carb_p90: float = 0.0

    fibre_mean: float = 0.0
    fibre_sd: float = 0.0
    fibre_p10: float = 0.0
    fibre_p90: float = 0.0

    samples_count: int = Field(default=2000, description="Monte Carlo draws S")

    @property
    def nominal_interval(self) -> tuple[float, float]:
        """Nominal Bayesian credible interval [p10, p90] in kcal."""
        return self.kcal_p10, self.kcal_p90


class ConformalInterval(BaseModel):
    """Conformally calibrated prediction interval satisfying finite-sample coverage."""

    model_config = ConfigDict(from_attributes=True)

    y_hat: float = Field(..., description="Point estimate (kcal_mean)")
    sigma_hat: float = Field(..., description="Standard deviation (kcal_sd)")
    q_hat: float = Field(..., description="Empirical non-conformity factor from D_cal")
    lower: float = Field(..., description="Calibrated lower bound max(0, Y_hat - q_hat * sigma)")
    upper: float = Field(..., description="Calibrated upper bound Y_hat + q_hat * sigma")
    nominal_p10: float = Field(..., description="Nominal uncalibrated 10th percentile")
    nominal_p90: float = Field(..., description="Nominal uncalibrated 90th percentile")
    coverage_level: float = Field(default=0.90, description="Target coverage probability 1 - alpha")

    @property
    def interval(self) -> tuple[float, float]:
        """Calibrated interval tuple [lower, upper]."""
        return self.lower, self.upper


def _extract_item_nutrient_input(item: Any) -> MealItemNutrientInput:
    """Normalize input item into MealItemNutrientInput."""
    if isinstance(item, MealItemNutrientInput):
        return item
    if isinstance(item, dict):
        g_val = item.get("grams_mean")
        sd_val = item.get("grams_sd")
        g_mean = float(g_val) if g_val is not None else None
        g_sd = float(sd_val) if sd_val is not None else None
        mu_log = float(item.get("mu_log", 0.0) or 0.0)
        sigma2_log = float(item.get("sigma2_log", 0.0) or 0.0)

        if (mu_log == 0.0 and sigma2_log == 0.0) and (g_mean is not None and g_mean > 0.0):
            effective_sd = g_sd if (g_sd is not None and g_sd > 0.0) else (g_mean * 0.15)
            sigma2_log = math.log(1.0 + (effective_sd / g_mean) ** 2)
            mu_log = math.log(g_mean) - sigma2_log / 2.0
        elif (g_mean is None or g_mean <= 0.0) and (mu_log != 0.0 or sigma2_log != 0.0):
            s2_eff = max(sigma2_log, 1e-6)
            g_mean = math.exp(mu_log + s2_eff / 2.0)
            g_sd = math.sqrt(math.exp(2.0 * mu_log + s2_eff) * (math.exp(s2_eff) - 1.0))

        return MealItemNutrientInput(
            food_name=item.get("food_name", item.get("name", "dish")),
            mu_log=mu_log,
            sigma2_log=sigma2_log,
            kcal_per_100g=float(item.get("kcal_per_100g", 0.0) or 0.0),
            protein_g=float(item.get("protein_g", 0.0) or 0.0),
            fat_g=float(item.get("fat_g", 0.0) or 0.0),
            carb_g=float(item.get("carb_g", 0.0) or 0.0),
            fibre_g=float(item.get("fibre_g", 0.0) or 0.0),
            grams_mean=g_mean,
            grams_sd=g_sd,
        )

    food = getattr(item, "food", None)
    if food is not None:
        kcal_per_100g = getattr(food, "kcal_per_100g", 0.0)
        protein_g = getattr(food, "protein_g", 0.0)
        fat_g = getattr(food, "fat_g", 0.0)
        carb_g = getattr(food, "carb_g", 0.0)
        fibre_g = getattr(food, "fibre_g", 0.0)
    else:
        g_val = getattr(item, "grams_mean", None)
        k_val = getattr(item, "kcal_mean", None)
        if g_val and k_val and float(g_val) > 0:
            kcal_per_100g = (float(k_val) / float(g_val)) * 100.0
        else:
            kcal_per_100g = getattr(item, "kcal_per_100g", 100.0) or 100.0
        protein_g = getattr(item, "protein_g", 0.0)
        fat_g = getattr(item, "fat_g", 0.0)
        carb_g = getattr(item, "carb_g", 0.0)
        fibre_g = getattr(item, "fibre_g", 0.0)
    g_mean = getattr(item, "grams_mean", None)
    g_sd = getattr(item, "grams_sd", None)
    mu_log = float(getattr(item, "mu_log", 0.0) or 0.0)
    sigma2_log = float(getattr(item, "sigma2_log", 0.0) or 0.0)

    # Bidirectional log-normal parameter synchronization
    g_mean_val = float(g_mean) if g_mean is not None else None
    g_sd_val = float(g_sd) if g_sd is not None else None

    if (mu_log == 0.0 and sigma2_log == 0.0) and (g_mean_val is not None and g_mean_val > 0.0):
        effective_sd = g_sd_val if (g_sd_val is not None and g_sd_val > 0.0) else (g_mean_val * 0.15)
        sigma2_log = math.log(1.0 + (effective_sd / g_mean_val) ** 2)
        mu_log = math.log(g_mean_val) - sigma2_log / 2.0
    elif (g_mean_val is None or g_mean_val <= 0.0) and (mu_log != 0.0 or sigma2_log != 0.0):
        s2_eff = max(sigma2_log, 1e-6)
        g_mean_val = math.exp(mu_log + s2_eff / 2.0)
        g_sd_val = math.sqrt(math.exp(2.0 * mu_log + s2_eff) * (math.exp(s2_eff) - 1.0))

    return MealItemNutrientInput(
        food_name=getattr(item, "dish_label_raw", getattr(item, "food_name", "dish")),
        mu_log=mu_log,
        sigma2_log=sigma2_log,
        kcal_per_100g=float(kcal_per_100g or 0.0),
        protein_g=float(protein_g or 0.0),
        fat_g=float(fat_g or 0.0),
        carb_g=float(carb_g or 0.0),
        fibre_g=float(fibre_g or 0.0),
        grams_mean=g_mean_val,
        grams_sd=g_sd_val,
    )


def compute_monte_carlo_total(
    items: Sequence[Any],
    num_samples: int = 2000,
    seed: Optional[int] = None,
) -> MealTotalResult:
    """Vectorized Monte Carlo nutrient compositor (§6.1.3).

    Performs S = 2000 log-normal draws for each meal component, evaluates
    linear calories and macronutrients, and aggregates total distribution.

    Args:
        items: List of meal items (dicts, MealItemNutrientInput, or ORM MealItem).
        num_samples: Number of Monte Carlo draws S (default 2000).
        seed: Optional RNG seed for deterministic reproducibility.

    Returns:
        MealTotalResult with empirical means, standard deviations, and [p10, p90] intervals.
    """
    if not items:
        return MealTotalResult(
            kcal_mean=0.0,
            kcal_sd=0.0,
            kcal_p10=0.0,
            kcal_p90=0.0,
            samples_count=num_samples,
        )

    parsed_items = [_extract_item_nutrient_input(it) for it in items]
    num_items = len(parsed_items)

    rng = np.random.default_rng(seed)

    mu_vec = np.array([it.mu_log for it in parsed_items], dtype=np.float64)[:, np.newaxis]
    sigma_vec = np.array(
        [math.sqrt(max(it.sigma2_log, 1e-6)) for it in parsed_items],
        dtype=np.float64,
    )[:, np.newaxis]

    # Standard normal draws -> lognormal grams draws (M items x S samples)
    z = rng.standard_normal(size=(num_items, num_samples))
    G = np.exp(mu_vec + sigma_vec * z)

    # Nutrient multiplier matrices (per gram)
    c_kcal = np.array([it.kcal_per_100g / 100.0 for it in parsed_items], dtype=np.float64)
    c_protein = np.array([(it.protein_g or 0.0) / 100.0 for it in parsed_items], dtype=np.float64)
    c_fat = np.array([(it.fat_g or 0.0) / 100.0 for it in parsed_items], dtype=np.float64)
    c_carb = np.array([(it.carb_g or 0.0) / 100.0 for it in parsed_items], dtype=np.float64)
    c_fibre = np.array([(it.fibre_g or 0.0) / 100.0 for it in parsed_items], dtype=np.float64)

    # Combined coefficient matrix shape: (5, M)
    C = np.vstack([c_kcal, c_protein, c_fat, c_carb, c_fibre])

    # Vectorized composition: (5, M) @ (M, S) -> (5, S)
    Nutrients = C @ G

    kcal_draws = Nutrients[0]
    protein_draws = Nutrients[1]
    fat_draws = Nutrients[2]
    carb_draws = Nutrients[3]
    fibre_draws = Nutrients[4]

    return MealTotalResult(
        kcal_mean=float(np.mean(kcal_draws)),
        kcal_sd=float(np.std(kcal_draws, ddof=1)),
        kcal_p10=float(np.percentile(kcal_draws, 10)),
        kcal_p90=float(np.percentile(kcal_draws, 90)),

        protein_mean=float(np.mean(protein_draws)),
        protein_sd=float(np.std(protein_draws, ddof=1)),
        protein_p10=float(np.percentile(protein_draws, 10)),
        protein_p90=float(np.percentile(protein_draws, 90)),

        fat_mean=float(np.mean(fat_draws)),
        fat_sd=float(np.std(fat_draws, ddof=1)),
        fat_p10=float(np.percentile(fat_draws, 10)),
        fat_p90=float(np.percentile(fat_draws, 90)),

        carb_mean=float(np.mean(carb_draws)),
        carb_sd=float(np.std(carb_draws, ddof=1)),
        carb_p10=float(np.percentile(carb_draws, 10)),
        carb_p90=float(np.percentile(carb_draws, 90)),

        fibre_mean=float(np.mean(fibre_draws)),
        fibre_sd=float(np.std(fibre_draws, ddof=1)),
        fibre_p10=float(np.percentile(fibre_draws, 10)),
        fibre_p90=float(np.percentile(fibre_draws, 90)),

        samples_count=num_samples,
    )


def compute_analytical_expectations(items: Sequence[Any]) -> dict[str, float]:
    """Calculate exact analytical closed-form expectation and standard deviation.

    For independent log-normal item masses:
        E[G_i] = exp(mu_i + sigma_i^2 / 2)
        Var(G_i) = exp(2*mu_i + sigma_i^2) * (exp(sigma_i^2) - 1)
        E[Total kcal] = sum_i c_i * E[G_i]
        SD[Total kcal] = sqrt(sum_i c_i^2 * Var(G_i))
    """
    parsed = [_extract_item_nutrient_input(it) for it in items]
    mean_total = 0.0
    var_total = 0.0

    for it in parsed:
        c_i = it.kcal_per_100g / 100.0
        s2 = max(it.sigma2_log, 1e-6)
        mean_g = math.exp(it.mu_log + s2 / 2.0)
        var_g = math.exp(2.0 * it.mu_log + s2) * (math.exp(s2) - 1.0)

        mean_total += c_i * mean_g
        var_total += (c_i ** 2) * var_g

    return {
        "kcal_mean": mean_total,
        "kcal_sd": math.sqrt(var_total),
    }


def compute_analytical_total(items: Sequence[Any]) -> MealTotalResult:
    """Exact closed-form analytical nutrient compositor (p01.md §4.1).

    Computes total meal expectation and variance analytically in sub-millisecond
    latency (< 10 microseconds) with zero sampling variance:
        E[Total] = sum_i c_i * E[G_i]
        Var[Total] = sum_i c_i^2 * Var(G_i)
        SD[Total] = sqrt(Var[Total])
        [p10, p90] = [max(0, E - 1.28155 * SD), E + 1.28155 * SD]
    """
    if not items:
        return MealTotalResult(
            kcal_mean=0.0,
            kcal_sd=0.0,
            kcal_p10=0.0,
            kcal_p90=0.0,
            protein_mean=0.0,
            protein_sd=0.0,
            protein_p10=0.0,
            protein_p90=0.0,
            fat_mean=0.0,
            fat_sd=0.0,
            fat_p10=0.0,
            fat_p90=0.0,
            carb_mean=0.0,
            carb_sd=0.0,
            carb_p10=0.0,
            carb_p90=0.0,
            fibre_mean=0.0,
            fibre_sd=0.0,
            fibre_p10=0.0,
            fibre_p90=0.0,
            samples_count=0,
        )

    parsed = [_extract_item_nutrient_input(it) for it in items]

    kcal_sum = 0.0
    kcal_var = 0.0
    protein_sum = 0.0
    protein_var = 0.0
    fat_sum = 0.0
    fat_var = 0.0
    carb_sum = 0.0
    carb_var = 0.0
    fibre_sum = 0.0
    fibre_var = 0.0

    for it in parsed:
        if it.grams_mean is not None and it.grams_mean > 0:
            mean_g = float(it.grams_mean)
            sd_g = float(it.grams_sd) if it.grams_sd is not None else mean_g * 0.15
        elif it.mu_log != 0.0 or it.sigma2_log != 0.0:
            s2 = max(it.sigma2_log, 1e-6)
            mean_g = math.exp(it.mu_log + s2 / 2.0)
            var_g = math.exp(2.0 * it.mu_log + s2) * (math.exp(s2) - 1.0)
            sd_g = math.sqrt(max(0.0, var_g))
        elif it.grams_mean is not None:
            mean_g = float(it.grams_mean)
            sd_g = float(it.grams_sd or 0.0)
        else:
            mean_g = 0.0
            sd_g = 0.0

        r_mean = mean_g / 100.0
        r_sd = sd_g / 100.0

        c_kcal = float(it.kcal_per_100g or 0.0)
        c_protein = float(it.protein_g or 0.0)
        c_fat = float(it.fat_g or 0.0)
        c_carb = float(it.carb_g or 0.0)
        c_fibre = float(it.fibre_g or 0.0)

        kcal_sum += c_kcal * r_mean
        kcal_var += (c_kcal * r_sd) ** 2

        protein_sum += c_protein * r_mean
        protein_var += (c_protein * r_sd) ** 2

        fat_sum += c_fat * r_mean
        fat_var += (c_fat * r_sd) ** 2

        carb_sum += c_carb * r_mean
        carb_var += (c_carb * r_sd) ** 2

        fibre_sum += c_fibre * r_mean
        fibre_var += (c_fibre * r_sd) ** 2

    # Standard normal quantile for 10th and 90th percentile: z_0.90 = 1.28155
    Z_90 = 1.28155

    kcal_mean = round(kcal_sum, 1)
    kcal_sd = round(math.sqrt(kcal_var), 1)
    kcal_p10 = max(0.0, round(kcal_mean - Z_90 * kcal_sd, 1))
    kcal_p90 = round(kcal_mean + Z_90 * kcal_sd, 1)

    protein_mean = round(protein_sum, 1)
    protein_sd = round(math.sqrt(protein_var), 1)
    protein_p10 = max(0.0, round(protein_mean - Z_90 * protein_sd, 1))
    protein_p90 = round(protein_mean + Z_90 * protein_sd, 1)

    fat_mean = round(fat_sum, 1)
    fat_sd = round(math.sqrt(fat_var), 1)
    fat_p10 = max(0.0, round(fat_mean - Z_90 * fat_sd, 1))
    fat_p90 = round(fat_mean + Z_90 * fat_sd, 1)

    carb_mean = round(carb_sum, 1)
    carb_sd = round(math.sqrt(carb_var), 1)
    carb_p10 = max(0.0, round(carb_mean - Z_90 * carb_sd, 1))
    carb_p90 = round(carb_mean + Z_90 * carb_sd, 1)

    fibre_mean = round(fibre_sum, 1)
    fibre_sd = round(math.sqrt(fibre_var), 1)
    fibre_p10 = max(0.0, round(fibre_mean - Z_90 * fibre_sd, 1))
    fibre_p90 = round(fibre_mean + Z_90 * fibre_sd, 1)

    return MealTotalResult(
        kcal_mean=kcal_mean,
        kcal_sd=kcal_sd,
        kcal_p10=kcal_p10,
        kcal_p90=kcal_p90,
        protein_mean=protein_mean,
        protein_sd=protein_sd,
        protein_p10=protein_p10,
        protein_p90=protein_p90,
        fat_mean=fat_mean,
        fat_sd=fat_sd,
        fat_p10=fat_p10,
        fat_p90=fat_p90,
        carb_mean=carb_mean,
        carb_sd=carb_sd,
        carb_p10=carb_p10,
        carb_p90=carb_p90,
        fibre_mean=fibre_mean,
        fibre_sd=fibre_sd,
        fibre_p10=fibre_p10,
        fibre_p90=fibre_p90,
        samples_count=0,
    )


def compute_conformal_q_hat(
    calibration_residuals: Sequence[float],
    alpha: float = 0.10,
) -> float:
    """Compute empirical non-conformity factor q_hat on calibration split D_cal (§13.3).

    Formula:
        R_i = |Y_{i, true} - Y_hat_i| / sigma_hat_i
        q_{1-alpha} = Quantile({R_i}, ceil((N + 1) * (1 - alpha)) / N)
    """
    n = len(calibration_residuals)
    if n == 0:
        return 1.645  # Default nominal standard Gaussian 90% factor

    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / float(n))
    return float(np.quantile(calibration_residuals, level))


def apply_conformal_calibration(
    total_result: MealTotalResult,
    q_hat: float,
    coverage_level: float = 0.90,
) -> ConformalInterval:
    """Apply split-conformal calibration factor q_hat to Monte Carlo estimate (§13.3).

    Constructs conformal interval:
        [p10, p90]_calibrated = [max(0, Y_hat - q_hat * sigma_hat), Y_hat + q_hat * sigma_hat]
    """
    y_hat = total_result.kcal_mean
    sigma_hat = total_result.kcal_sd
    lower = max(0.0, y_hat - float(q_hat) * sigma_hat)
    upper = y_hat + float(q_hat) * sigma_hat

    return ConformalInterval(
        y_hat=y_hat,
        sigma_hat=sigma_hat,
        q_hat=float(q_hat),
        lower=lower,
        upper=upper,
        nominal_p10=total_result.kcal_p10,
        nominal_p90=total_result.kcal_p90,
        coverage_level=coverage_level,
    )
