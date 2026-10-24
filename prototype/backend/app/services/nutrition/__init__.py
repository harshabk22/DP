"""Monte Carlo nutrient composition and conformal prediction package.

Adheres to MESSMATE_SPEC.md §6.1.3 and §13.3.
"""

from app.services.nutrition.monte_carlo import (
    ConformalInterval,
    MealItemNutrientInput,
    MealTotalResult,
    apply_conformal_calibration,
    compute_analytical_expectations,
    compute_analytical_total,
    compute_conformal_q_hat,
    compute_monte_carlo_total,
)

__all__ = [
    "MealItemNutrientInput",
    "MealTotalResult",
    "ConformalInterval",
    "compute_monte_carlo_total",
    "compute_analytical_total",
    "compute_analytical_expectations",
    "apply_conformal_calibration",
    "compute_conformal_q_hat",
]
