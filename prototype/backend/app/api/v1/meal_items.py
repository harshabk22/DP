"""Meal item modification endpoints for refill chips and manual edits (Invariant I5)."""

from datetime import datetime, timezone
import math
from typing import Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.meal import Meal, MealItem
from app.models.prior import UnitPrior
from app.models.question import Correction
from app.models.user import User
from app.schemas.meal import MealItemDetail, MealItemUpdate
from app.services.conformal.engine import ConformalCalibrationService
from app.services.nutrition.monte_carlo import compute_analytical_total
from app.services.priors.updater import calculate_portion_distribution
from app.tasks.meal_pipeline import ARCHETYPE_MASS_G, ARCHETYPE_VOLUMES_ML

router = APIRouter(prefix="/meal-items", tags=["meal-items"])


@router.patch(
    "/{item_id}",
    response_model=MealItemDetail,
    summary="Update a meal item via refill chip or manual edit (Invariant I5)",
)
def patch_meal_item(
    item_id: uuid.UUID,
    item_in: MealItemUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MealItemDetail:
    """Apply manual correction or refill chip adjustment.

    Strictly appends an immutable audit log to corrections (Invariant I5)
    and updates the meal's Monte Carlo distribution moments.
    """
    meal_item = db.query(MealItem).filter(MealItem.id == item_id).first()
    if not meal_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meal item not found.",
        )

    # Verify authorization
    meal = meal_item.meal
    if meal.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this meal item.",
        )

    # Log immutable correction for Invariant I5
    field_modified = "unit_count"
    old_val = {
        "unit_count": meal_item.unit_count,
        "unit": meal_item.unit,
        "grams": meal_item.grams_mean,
        "is_refill": meal_item.is_refill,
    }
    new_val = {}

    if item_in.unit_count is not None:
        new_val["unit_count"] = item_in.unit_count
        field_modified = "unit_count"
    if item_in.unit is not None:
        new_val["unit"] = item_in.unit
        field_modified = "unit"
    if item_in.grams is not None:
        new_val["grams"] = item_in.grams
        field_modified = "grams"
    if item_in.oil_level is not None:
        new_val["oil_level"] = item_in.oil_level
        field_modified = "oil_level"
    if item_in.is_refill is not None:
        new_val["is_refill"] = item_in.is_refill

    correction = Correction(
        user_id=current_user.id,
        meal_item_id=meal_item.id,
        field=field_modified,
        old_value=old_val,
        new_value=new_val,
        source="manual_edit",
    )
    db.add(correction)

    # Apply changes
    if item_in.unit_count is not None:
        meal_item.unit_count = item_in.unit_count
    if item_in.unit is not None:
        meal_item.unit = item_in.unit
    if item_in.is_refill is not None:
        meal_item.is_refill = item_in.is_refill

    food = meal_item.food

    if item_in.grams is not None:
        meal_item.grams_mean = float(item_in.grams)
        meal_item.grams_sd = 2.0
        meal_item.portion_source = "user"
        meal_item.mu_log = math.log(max(meal_item.grams_mean, 1e-6))
        meal_item.sigma2_log = (meal_item.grams_sd / max(meal_item.grams_mean, 1e-6)) ** 2
        if food is not None:
            ratio = meal_item.grams_mean / 100.0
            meal_item.kcal_mean = float(food.kcal_per_100g) * ratio
            meal_item.kcal_sd = float(food.kcal_per_100g) * (meal_item.grams_sd / 100.0)
        else:
            ratio = meal_item.grams_mean / 100.0
            meal_item.kcal_mean = 100.0 * ratio
            meal_item.kcal_sd = 100.0 * (meal_item.grams_sd / 100.0)
    elif item_in.unit_count is not None or item_in.unit is not None:
        # Re-calculate portion and calories from prior / archetypes
        unit = meal_item.unit or "standard_bowl"
        count = max(float(meal_item.unit_count or 1.0), 0.1)
        rho = float(food.density_g_per_ml) if food else 1.0
        is_vessel = unit in ARCHETYPE_VOLUMES_ML

        prior = (
            db.query(UnitPrior)
            .filter(
                UnitPrior.user_id == current_user.id,
                UnitPrior.unit == unit,
                UnitPrior.context == meal.context,
            )
            .first()
        )

        if prior:
            theta_um = float(prior.mu_log)
            sigma2_um = float(prior.sigma2_log)
        elif is_vessel:
            theta_um = math.log(ARCHETYPE_VOLUMES_ML.get(unit, 350.0))
            sigma2_um = 0.04
        else:
            theta_um = math.log(ARCHETYPE_MASS_G.get(unit, 150.0))
            sigma2_um = 0.04

        portion_est = calculate_portion_distribution(
            theta_um=theta_um,
            sigma2_um=sigma2_um,
            rho_d=rho,
            n=count,
            kcal_per_100g=float(food.kcal_per_100g) if food else 100.0,
            is_vessel=is_vessel,
        )

        meal_item.grams_mean = portion_est.grams_mean
        meal_item.grams_sd = portion_est.grams_sd
        meal_item.kcal_mean = portion_est.kcal_mean
        meal_item.kcal_sd = portion_est.kcal_sd
        meal_item.mu_log = portion_est.mu_log
        meal_item.sigma2_log = portion_est.sigma2_log
        meal_item.portion_source = "user"

    # Re-evaluate meal analytical closed-form total (p01.md §4.1)
    analytical_res = compute_analytical_total(meal.items)
    meal.kcal_mean = analytical_res.kcal_mean
    meal.kcal_sd = analytical_res.kcal_sd
    
    # Enforce conformal prediction calibration bounds [p10, p90] on refill updates
    conf_bounds = ConformalCalibrationService.calibrate_meal_bounds(
        kcal_mean=analytical_res.kcal_mean,
        kcal_sd=analytical_res.kcal_sd,
    )
    meal.kcal_p10 = conf_bounds.kcal_p10
    meal.kcal_p90 = conf_bounds.kcal_p90

    db.commit()
    db.refresh(meal_item)

    return MealItemDetail(
        id=meal_item.id,
        dish_label=meal_item.dish_label_raw,
        food_id=meal_item.food_id,
        resolution_tier=meal_item.resolution_tier,
        unit=meal_item.unit,
        unit_count=meal_item.unit_count,
        grams_mean=meal_item.grams_mean,
        grams_sd=meal_item.grams_sd,
        is_refill=meal_item.is_refill,
        vlm_grams_raw=meal_item.vlm_grams_raw,
        kcal_mean=meal_item.kcal_mean,
        kcal_sd=meal_item.kcal_sd,
        portion_source=meal_item.portion_source,
        ask_value=meal_item.ask_value,
    )
