"""User profile, hard deletion, and data export endpoints."""

from datetime import datetime, timezone
import hashlib
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.safety import (
    DISCLAIMER_TEXT,
    calculate_bmr,
    clamp_goal_target,
    get_user_daily_targets,
    validate_weight_loss_rate,
)
import math
from app.models.event import Event
from app.models.prior import UnitPrior, UnitPriorHistory
from app.models.triggers import hard_delete_session
from app.models.user import Consent, Profile, User
from app.schemas.user import ProfileResponse, ProfileUpdate, TablewareOnboardingRequest

router = APIRouter(prefix="/me", tags=["users"])


def _build_profile_response(profile: Profile, target_kcal: Any = None) -> ProfileResponse:
    """Compute BMR and format profile response with Invariant I8 safeguards."""
    bmr = calculate_bmr(
        birth_year=profile.birth_year,
        sex=profile.sex,
        weight_kg=profile.weight_kg,
        height_cm=profile.height_cm,
    )
    dyn_target_kcal, _, _, _ = get_user_daily_targets(profile)
    effective_target = target_kcal if target_kcal is not None else dyn_target_kcal
    clamped_target = clamp_goal_target(effective_target, bmr)

    return ProfileResponse(
        user_id=profile.user_id,
        birth_year=profile.birth_year,
        sex=profile.sex,
        height_cm=profile.height_cm,
        weight_kg=profile.weight_kg,
        activity_level=profile.activity_level,
        goal=profile.goal,
        target_kcal=effective_target,
        clamped_target_kcal=clamped_target,
        bmr=bmr,
        diet_type=profile.diet_type,
        allergies=profile.allergies or [],
        conditions=profile.conditions or [],
        region=profile.region,
        locale=profile.locale or "en-IN",
        cafeteria_id=profile.cafeteria_id,
        disclaimer=DISCLAIMER_TEXT,
    )


@router.get(
    "/profile",
    response_model=ProfileResponse,
    summary="Get authenticated user profile and BMR",
)
def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProfileResponse:
    """Retrieve profile including Mifflin–St Jeor BMR calculation."""
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    if not profile:
        profile = Profile(user_id=current_user.id, birth_year=2000, sex="other")
        db.add(profile)
        db.commit()
        db.refresh(profile)

    return _build_profile_response(profile)


@router.put(
    "/profile",
    response_model=ProfileResponse,
    summary="Update profile with age validation and BMR clamping (Invariants I7, I8)",
)
def update_profile(
    profile_in: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProfileResponse:
    """Update profile attributes and clamp target calories to at or above BMR."""
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    if not profile:
        profile = Profile(user_id=current_user.id, birth_year=2000, sex="other")
        db.add(profile)

    # Invariant I7: Check birth year threshold if provided
    if profile_in.birth_year is not None:
        current_year = datetime.now(timezone.utc).year
        if profile_in.birth_year > (current_year - 18):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invariant I7 violated: Users must be at least 18 years old. Birth year {profile_in.birth_year} implies age < 18.",
            )
        profile.birth_year = profile_in.birth_year

    if profile_in.sex is not None:
        profile.sex = profile_in.sex
    if profile_in.height_cm is not None:
        profile.height_cm = profile_in.height_cm
    if profile_in.weight_kg is not None:
        profile.weight_kg = profile_in.weight_kg
    if profile_in.activity_level is not None:
        profile.activity_level = profile_in.activity_level
    if profile_in.goal is not None:
        profile.goal = profile_in.goal
    if profile_in.diet_type is not None:
        profile.diet_type = profile_in.diet_type
    if profile_in.allergies is not None:
        profile.allergies = profile_in.allergies
    if profile_in.conditions is not None:
        profile.conditions = profile_in.conditions
    if profile_in.region is not None:
        profile.region = profile_in.region
    if profile_in.locale is not None:
        profile.locale = profile_in.locale
    if profile_in.cafeteria_id is not None:
        profile.cafeteria_id = profile_in.cafeteria_id

    db.commit()
    db.refresh(profile)

    return _build_profile_response(profile, target_kcal=profile_in.target_kcal)


@router.delete(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Hard delete user account and all personal data (GDPR/DPDP erasure)",
)
def delete_user(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Execute complete hard deletion with cascade and Invariant I5 trigger bypass."""
    with hard_delete_session(db):
        db.delete(current_user)
        db.commit()


@router.get(
    "/export",
    summary="Data portability export bundle (§12)",
)
def export_user_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Export complete de-identified personal data archive."""
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    consents = db.query(Consent).filter(Consent.user_id == current_user.id).all()

    # Anonymous participant hash for scientific research export
    anon_id = f"u_{hashlib.sha256(str(current_user.id).encode()).hexdigest()[:8]}"

    meals_data = []
    for meal in current_user.meals:
        meals_data.append(
            {
                "meal_id": str(meal.id),
                "captured_at": meal.captured_at.isoformat(),
                "local_date": meal.local_date.isoformat(),
                "status": meal.status,
                "context": meal.context,
                "total": {
                    "kcal_mean": meal.kcal_mean,
                    "kcal_sd": meal.kcal_sd,
                    "kcal_p10": meal.kcal_p10,
                    "kcal_p90": meal.kcal_p90,
                },
                "items": [
                    {
                        "item_id": str(item.id),
                        "dish_label": item.dish_label_raw,
                        "unit": item.unit,
                        "unit_count": item.unit_count,
                        "grams_mean": item.grams_mean,
                        "kcal_mean": item.kcal_mean,
                    }
                    for item in meal.items
                ],
            }
        )

    corrections_data = [
        {
            "id": str(c.id),
            "field": c.field,
            "old_value": c.old_value,
            "new_value": c.new_value,
            "source": c.source,
            "created_at": c.created_at.isoformat(),
        }
        for c in current_user.corrections
    ]

    return {
        "anonymous_participant_id": anon_id,
        "email": current_user.email,
        "study_arm": current_user.study_arm,
        "created_at": current_user.created_at.isoformat(),
        "profile": {
            "birth_year": profile.birth_year if profile else None,
            "sex": profile.sex if profile else None,
            "height_cm": profile.height_cm if profile else None,
            "weight_kg": profile.weight_kg if profile else None,
            "goal": profile.goal if profile else None,
            "diet_type": profile.diet_type if profile else None,
        },
        "consents": [
            {
                "purpose": c.purpose,
                "granted": c.granted,
                "notice_version": c.notice_version,
                "granted_at": c.granted_at.isoformat(),
            }
            for c in consents
        ],
        "meals": meals_data,
        "corrections": corrections_data,
    }


@router.post(
    "/onboarding/tableware",
    summary="Day 1 Visual Kitchen Mirror: calibrate tableware silhouettes (§6.2)",
)
def calibrate_tableware(
    onboarding_in: TablewareOnboardingRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Seed user tableware unit priors (mu_0 = ln(volume), sigma2_0 = 0.05, n_obs = 1)."""
    now = datetime.now(timezone.utc)
    seeded = []

    for item in onboarding_in.selections:
        vol = max(float(item.volume_ml), 1.0)
        mu_log = math.log(vol)
        sigma2_log = 0.05  # Informative setup prior

        prior = (
            db.query(UnitPrior)
            .filter(
                UnitPrior.user_id == current_user.id,
                UnitPrior.unit == item.unit,
                UnitPrior.context == onboarding_in.context,
            )
            .first()
        )

        if prior:
            prior.mu_log = mu_log
            prior.sigma2_log = sigma2_log
            prior.n_obs = 1
            prior.consecutive_rejects = 0
            prior.last_reject_sign = 0
            prior.updated_at = now
        else:
            prior = UnitPrior(
                user_id=current_user.id,
                unit=item.unit,
                context=onboarding_in.context,
                mu_log=mu_log,
                sigma2_log=sigma2_log,
                n_obs=1,
                consecutive_rejects=0,
                last_reject_sign=0,
                updated_at=now,
            )
            db.add(prior)

        # Log to unit prior history
        history = UnitPriorHistory(
            user_id=current_user.id,
            unit=item.unit,
            context=onboarding_in.context,
            mu_log=mu_log,
            sigma2_log=sigma2_log,
            n_obs=1,
            created_at=now,
        )
        db.add(history)
        seeded.append({"unit": item.unit, "volume_ml": vol, "mu_log": round(mu_log, 4)})

    # Log telemetry event
    db.add(
        Event(
            user_id=current_user.id,
            type="tableware_onboarding_completed",
            meta={"seeded_count": len(seeded), "context": onboarding_in.context},
            created_at=now,
        )
    )

    db.commit()
    return {
        "status": "success",
        "message": f"Successfully calibrated {len(seeded)} tableware archetypes with informative prior (sigma2=0.05).",
        "seeded_units": seeded,
    }
