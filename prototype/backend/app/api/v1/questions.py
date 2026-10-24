"""Question answering endpoints enforcing Invariants I3, I4, and I5."""

from datetime import datetime, timezone
import math
from typing import Any, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.food import Food
from app.models.meal import Meal, MealItem
from app.models.prior import UnitPrior, UnitPriorHistory
from app.models.question import Correction, Question
from app.models.user import User
from app.schemas.question import QuestionAnswerRequest, QuestionResponse
from app.services.conformal.engine import ConformalCalibrationService
from app.services.nutrition.monte_carlo import compute_analytical_total
from app.services.priors.updater import (
    calculate_portion_distribution,
    update_unit_prior,
)
from app.services.resolution.engine import ResolutionService
from app.tasks.meal_pipeline import ARCHETYPE_MASS_G, ARCHETYPE_VOLUMES_ML

router = APIRouter(prefix="/meals", tags=["questions"])


def _extract_numeric_value(answer_payload: Any) -> Optional[float]:
    """Extract discrete count or numeric choice from answer payload."""
    if answer_payload is None:
        return None
    if isinstance(answer_payload, (int, float)):
        return float(answer_payload)
    if isinstance(answer_payload, dict):
        val = answer_payload.get("value", answer_payload.get("val", answer_payload.get("count")))
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    if isinstance(answer_payload, str):
        # Handle cases like "2" or "4+"
        clean = answer_payload.strip().rstrip("+")
        try:
            return float(clean)
        except ValueError:
            pass
    return None


@router.post(
    "/{meal_id}/answer",
    response_model=QuestionResponse,
    summary="Answer or skip an allocated meal question (Invariants I3, I4, I5)",
)
def answer_question(
    meal_id: uuid.UUID,
    answer_in: QuestionAnswerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> QuestionResponse:
    """Submit discrete tap response to a pending meal question.

    Updates personal Bayesian tableware prior (if personal_home context) and
    appends an immutable record to the corrections table (Invariant I5).
    """
    meal = (
        db.query(Meal)
        .filter(Meal.id == meal_id, Meal.user_id == current_user.id)
        .first()
    )
    if not meal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meal not found.",
        )

    question = db.query(Question).filter(Question.meal_id == meal.id).first()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pending question found for this meal.",
        )

    if question.answered_at is not None or question.skipped:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question has already been answered or skipped.",
        )

    now = datetime.now(timezone.utc)

    # Handle Skip (Invariant I4: 1-tap frictionless dismissal)
    if answer_in.skipped:
        question.skipped = True
        question.answered_at = now
        db.commit()
        db.refresh(question)
        return QuestionResponse.model_validate(question)

    # Validate non-skipped answer payload
    if answer_in.answer is None or str(answer_in.answer).strip() == "":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invariant I4: Non-skipped question requires a discrete tap answer.",
        )

    # Process Answer
    target_item = (
        db.query(MealItem).filter(MealItem.id == question.meal_item_id).first()
        if question.meal_item_id
        else None
    )

    numeric_val = _extract_numeric_value(answer_in.answer)

    # Polymorphic Handling 1: Dish Disambiguation Question
    if target_item and question.question_type == "dish_disambiguation":
        chosen_dish = str(answer_in.answer).strip()
        if chosen_dish:
            old_dish = target_item.dish_label_raw
            old_food_id = target_item.food_id
            resolved = ResolutionService(db).resolve(chosen_dish)
            target_item.dish_label_raw = chosen_dish
            target_item.food_id = resolved.food_id
            target_item.resolution_tier = resolved.resolution_tier
            target_item.dish_uncertain = False

            db.add(Correction(
                user_id=current_user.id,
                meal_item_id=target_item.id,
                field="dish",
                old_value={"dish": old_dish, "food_id": old_food_id},
                new_value={"dish": chosen_dish, "food_id": resolved.food_id},
                source="question",
            ))

            food = db.query(Food).filter(Food.id == resolved.food_id).first() if resolved.food_id else None
            kcal_per_g = (float(food.kcal_per_100g) / 100.0) if food else 1.35
            target_item.kcal_mean = round((target_item.grams_mean or 150.0) * kcal_per_g, 1)
            target_item.kcal_sd = round((target_item.grams_sd or 20.0) * kcal_per_g, 1)
            target_item.portion_source = "user"
            target_item.ask_value = 0.0

            res = compute_analytical_total(meal.items)
            meal.kcal_mean = res.kcal_mean
            meal.kcal_sd = res.kcal_sd
            conf_bounds = ConformalCalibrationService.calibrate_meal_bounds(
                kcal_mean=res.kcal_mean,
                kcal_sd=res.kcal_sd,
            )
            meal.kcal_p10 = conf_bounds.kcal_p10
            meal.kcal_p90 = conf_bounds.kcal_p90

    # Polymorphic Handling 2: Oil Level Question
    elif target_item and question.question_type == "oil_level":
        oil_choice = str(answer_in.answer).lower().strip()
        oil_multipliers = {"light": 0.90, "normal": 1.0, "extra": 1.15}
        mult = oil_multipliers.get(oil_choice, 1.0)
        old_kcal = target_item.kcal_mean
        target_item.kcal_mean = round(target_item.kcal_mean * mult, 1)
        target_item.ask_value = 0.0
        db.add(Correction(
            user_id=current_user.id,
            meal_item_id=target_item.id,
            field="oil_level",
            old_value={"kcal_mean": old_kcal},
            new_value={"kcal_mean": target_item.kcal_mean, "oil_level": oil_choice},
            source="question",
        ))
        res = compute_analytical_total(meal.items)
        meal.kcal_mean = res.kcal_mean
        meal.kcal_sd = res.kcal_sd
        conf_bounds = ConformalCalibrationService.calibrate_meal_bounds(
            kcal_mean=res.kcal_mean,
            kcal_sd=res.kcal_sd,
        )
        meal.kcal_p10 = conf_bounds.kcal_p10
        meal.kcal_p90 = conf_bounds.kcal_p90

    # Standard Unit Count / Unit Size Question
    elif target_item and numeric_val is not None:
        old_count = target_item.unit_count
        new_count = numeric_val

        # Invariant I5: Append immutable correction log
        correction = Correction(
            user_id=current_user.id,
            meal_item_id=target_item.id,
            field="unit_count",
            old_value={"unit_count": old_count, "unit": target_item.unit},
            new_value={"unit_count": new_count, "unit": target_item.unit},
            source="question",
        )
        db.add(correction)

        # Bayesian Tableware Prior Update (Context-gated: personal_home only)
        unit = target_item.unit or "standard_bowl"
        is_home = meal.context == "personal_home"
        food = target_item.food
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

        if not prior:
            init_vol = ARCHETYPE_VOLUMES_ML.get(unit, 350.0) if is_vessel else ARCHETYPE_MASS_G.get(unit, 150.0)
            prior = UnitPrior(
                user_id=current_user.id,
                unit=unit,
                context=meal.context,
                mu_log=math.log(init_vol),
                sigma2_log=0.05,
                n_obs=0,
                consecutive_rejects=0,
                last_reject_sign=0,
            )
            db.add(prior)
            db.flush()

        # Check Feast Shield status from meal model run
        is_feast_active = False
        for mr in meal.model_runs:
            if isinstance(mr.raw_responses, dict) and mr.raw_responses.get("feast_shield_active"):
                is_feast_active = True
                break

        prior_result = update_unit_prior(
            prior_mu=float(prior.mu_log),
            prior_s2=float(prior.sigma2_log),
            g_vlm=float(target_item.vlm_grams_raw or target_item.grams_mean or 150.0),
            n_user=new_count,
            rho_dish=rho,
            is_home=is_home,
            consecutive_rejects=prior.consecutive_rejects,
            last_reject_sign=prior.last_reject_sign,
            global_feast_shield=is_feast_active,
        )

        # Strike state must be persisted regardless of was_updated
        # to ensure Strike 1 and Strike 2 are tracked across meals!
        prior.consecutive_rejects = prior_result.consecutive_rejects
        prior.last_reject_sign = prior_result.last_reject_sign
        prior.updated_at = now

        if prior_result.was_updated:
            prior.mu_log = prior_result.mu_new
            prior.sigma2_log = prior_result.sigma2_new
            prior.n_obs += 1

            # Record trajectory in history table (§7)
            hist = UnitPriorHistory(
                user_id=current_user.id,
                unit=unit,
                context=meal.context,
                mu_log=prior_result.mu_new,
                sigma2_log=prior_result.sigma2_new,
                n_obs=prior.n_obs,
                triggered_by=meal.id,
            )
            db.add(hist)

        # Update meal item portion distribution moments
        target_item.unit_count = new_count
        portion_est = calculate_portion_distribution(
            theta_um=float(prior.mu_log),
            sigma2_um=float(prior.sigma2_log),
            rho_d=rho,
            n=new_count,
            kcal_per_100g=float(food.kcal_per_100g) if food else 100.0,
            is_vessel=is_vessel,
        )
        target_item.grams_mean = portion_est.grams_mean
        target_item.grams_sd = portion_est.grams_sd
        target_item.kcal_mean = portion_est.kcal_mean
        target_item.kcal_sd = portion_est.kcal_sd
        target_item.mu_log = portion_est.mu_log
        target_item.sigma2_log = portion_est.sigma2_log
        target_item.portion_source = "user"

        # Re-evaluate analytical closed-form total (p01.md §4.1)
        # and apply split-conformal calibration scaling (SPEC §13.3)
        res = compute_analytical_total(meal.items)
        meal.kcal_mean = res.kcal_mean
        meal.kcal_sd = res.kcal_sd
        conf_bounds = ConformalCalibrationService.calibrate_meal_bounds(
            kcal_mean=res.kcal_mean,
            kcal_sd=res.kcal_sd,
        )
        meal.kcal_p10 = conf_bounds.kcal_p10
        meal.kcal_p90 = conf_bounds.kcal_p90

    question.answer = answer_in.answer
    question.answered_at = now
    db.commit()
    db.refresh(question)

    return QuestionResponse.model_validate(question)


@router.post(
    "/questions/{question_id}/answer",
    response_model=QuestionResponse,
    summary="Answer question directly via question ID",
    include_in_schema=False,
)
def answer_question_by_id(
    question_id: uuid.UUID,
    answer_in: QuestionAnswerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> QuestionResponse:
    """Convenience alias allowing answering directly with question ID."""
    q = db.query(Question).filter(Question.id == question_id).first()
    if not q:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found.")
    return answer_question(meal_id=q.meal_id, answer_in=answer_in, current_user=current_user, db=db)
