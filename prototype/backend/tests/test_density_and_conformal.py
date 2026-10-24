"""Unit tests for Culinary Bulk Density and Split-Conformal Calibration Engines."""

import pytest
from app.services.density.engine import (
    CulinaryDensityService,
    DensityCategory,
)
from app.services.conformal.engine import (
    ConformalCalibrationService,
    DEFAULT_CONFORMAL_Q_90,
)


def test_density_service_classification():
    """Verify semantic regex routing across Indian culinary categories."""
    # Grains
    p_rice = CulinaryDensityService.resolve_density("Steamed White Rice")
    assert p_rice.category == DensityCategory.GRAINS
    assert p_rice.nominal_density == 0.84

    # Lentils / Dals
    p_dal = CulinaryDensityService.resolve_density("Tadka Toor Dal")
    assert p_dal.category == DensityCategory.LENTILS_DALS
    assert p_dal.nominal_density == 1.04

    # Curries / Gravies
    p_paneer = CulinaryDensityService.resolve_density("Paneer Butter Masala")
    assert p_paneer.category == DensityCategory.CURRIES_GRAVIES
    assert p_paneer.nominal_density == 1.02

    # Dry Sabzi
    p_bhindi = CulinaryDensityService.resolve_density("Bhindi Masala Fry")
    assert p_bhindi.category == DensityCategory.DRY_SABZI
    assert p_bhindi.nominal_density == 0.70

    # Salads / Greens
    p_salad = CulinaryDensityService.resolve_density("Cucumber Kachumber Salad")
    assert p_salad.category == DensityCategory.SALADS_GREENS
    assert p_salad.nominal_density == 0.28

    # Dairy
    p_curd = CulinaryDensityService.resolve_density("Fresh Curd Dahi")
    assert p_curd.category == DensityCategory.DAIRY_BEVERAGES
    assert p_curd.nominal_density == 1.04

    # Unknown / Generic fallback
    p_unknown = CulinaryDensityService.resolve_density("Exotic Mystery Mix")
    assert p_unknown.category == DensityCategory.GENERIC
    assert p_unknown.nominal_density == 1.00


def test_conformal_calibration_service_bounds():
    """Verify split-conformal calibration calculation and coverage expansion."""
    res = ConformalCalibrationService.calibrate_meal_bounds(
        kcal_mean=500.0,
        kcal_sd=50.0,
    )

    # With default q=1.842:
    # kcal_p10 = 500 - 1.842 * 50 = 407.9
    # kcal_p90 = 500 + 1.842 * 50 = 592.1
    assert res.y_hat == 500.0
    assert res.sigma_hat == 50.0
    assert res.q_hat == DEFAULT_CONFORMAL_Q_90
    assert res.kcal_p10 == pytest.approx(407.9, 0.1)
    assert res.kcal_p90 == pytest.approx(592.1, 0.1)

    # Verify conformal bounds are wider than nominal Gaussian bounds (Z=1.28155)
    # ensuring finite-sample distribution-free coverage
    assert res.kcal_p10 < res.nominal_p10
    assert res.kcal_p90 > res.nominal_p90
    assert res.coverage_guarantee == 0.90


def test_conformal_quantile_computation():
    """Verify finite-sample quantile calculation on calibration residuals."""
    residuals = [0.5, 0.8, 1.1, 1.3, 1.5, 1.7, 1.8, 2.0, 2.2, 2.5]
    q = ConformalCalibrationService.compute_q_hat(residuals, alpha=0.10)
    assert q >= 2.0
    assert isinstance(q, float)
