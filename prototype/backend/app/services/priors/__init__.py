"""Bayesian tableware prior calibration package.

Adheres to MESSMATE_SPEC.md §6.2 and Master Blueprint §3.
"""

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
    compute_lognormal_analytical_moments,
    evaluate_thali_joint_anomaly,
    update_thali_priors,
    update_unit_prior,
)

__all__ = [
    "SIGMA2_MIN",
    "INFLATED_SIGMA2",
    "GLOBAL_FEAST_SHIELD_FLAG",
    "PriorUpdateResult",
    "PortionEstimate",
    "ThaliShieldResult",
    "ThaliDishInput",
    "update_unit_prior",
    "update_thali_priors",
    "calculate_portion_distribution",
    "compute_dual_volume_estimators",
    "compute_lognormal_analytical_moments",
    "compute_conformal_bounds",
    "evaluate_thali_joint_anomaly",
]
