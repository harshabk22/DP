"""Safety guardrails and clinical health validation rules.

Enforces:
- Invariant I8: Mifflin-St Jeor BMR floor clamping, anti-shaming copy, non-clinical disclaimer.
- Invariant I7: Age threshold calculation.
- Safe weight-loss velocity checks (<1% body weight / week).
"""

from datetime import datetime, timezone
import re
from typing import Any, Optional

DISCLAIMER_TEXT = "Not medical advice; academic research prototype."

BANNED_SHAMING_PATTERNS = [
    r"\bover\s+budget\b",
    r"\byou\s+failed\b",
    r"\bstreak\s+lost\b",
    r"\bstreak\s+broken\b",
    r"\bbroken\s+streak\b",
    r"\blost\s+(?:your\s+)?streak\b",
    r"\bfailed\s+(?:your\s+)?goal\b",
    r"\bruined\b",
    r"\bcheat\s+(?:day|meal)\b",
    r"\bbad\s+job\b",
    r"\bguilt\b",
    r"\bpunish\w*\b",
]


def calculate_bmr(
    birth_year: int,
    sex: str,
    weight_kg: Optional[float] = None,
    height_cm: Optional[float] = None,
    current_year: Optional[int] = None,
) -> float:
    """Calculate Basal Metabolic Rate (BMR) via Mifflin–St Jeor with WHO default fallbacks.

    Formulas:
    - Male:   10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    - Female: 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    - Other:  10 * weight_kg + 6.25 * height_cm - 5 * age - 78 (midpoint)

    WHO default reference adult fallbacks:
    - Male: 70.0 kg, 175.0 cm
    - Female: 60.0 kg, 162.0 cm
    - Other: 65.0 kg, 168.0 cm
    """
    if current_year is None:
        current_year = datetime.now(timezone.utc).year

    age = max(1, current_year - birth_year)
    sex_clean = (sex or "other").strip().lower()

    # Apply WHO default fallbacks for missing physical anthropometrics
    if sex_clean == "male":
        eff_weight = weight_kg if weight_kg is not None and weight_kg > 0 else 70.0
        eff_height = height_cm if height_cm is not None and height_cm > 0 else 175.0
        s_offset = 5.0
    elif sex_clean == "female":
        eff_weight = weight_kg if weight_kg is not None and weight_kg > 0 else 60.0
        eff_height = height_cm if height_cm is not None and height_cm > 0 else 162.0
        s_offset = -161.0
    else:
        eff_weight = weight_kg if weight_kg is not None and weight_kg > 0 else 65.0
        eff_height = height_cm if height_cm is not None and height_cm > 0 else 168.0
        s_offset = -78.0

    bmr = (10.0 * eff_weight) + (6.25 * eff_height) - (5.0 * age) + s_offset
    return round(float(bmr), 1)


def clamp_goal_target(target_kcal: float, bmr: float) -> float:
    """Clamp target calorie goal to be at or above user's estimated BMR (Invariant I8)."""
    return round(max(float(target_kcal), float(bmr)), 1)


ACTIVITY_MULTIPLIERS: dict[str, float] = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

GOAL_ADJUSTMENTS: dict[str, float] = {
    "maintain": 0.0,
    "awareness": 0.0,
    "lose": -500.0,
    "gain": 500.0,
}


def get_user_daily_targets(profile: Any | None) -> tuple[float, float, float, float]:
    """Returns (target_kcal, target_protein_g, target_carb_g, target_fat_g).

    Calculates Mifflin-St Jeor BMR, applies activity multiplier and goal adjustments,
    and strictly enforces Invariant I8 (caloric target >= BMR).
    """
    if profile is None:
        return 2000.0, 75.0, 250.0, 65.0

    if isinstance(profile, dict):
        birth_year = profile.get("birth_year")
        sex = profile.get("sex")
        weight_kg = profile.get("weight_kg")
        height_cm = profile.get("height_cm")
        activity_level = profile.get("activity_level")
        goal = profile.get("goal")
        target_kcal_override = profile.get("target_kcal")
    else:
        birth_year = getattr(profile, "birth_year", None)
        sex = getattr(profile, "sex", None)
        weight_kg = getattr(profile, "weight_kg", None)
        height_cm = getattr(profile, "height_cm", None)
        activity_level = getattr(profile, "activity_level", None)
        goal = getattr(profile, "goal", None)
        target_kcal_override = getattr(profile, "target_kcal", None)

    if birth_year is None:
        return 2000.0, 75.0, 250.0, 65.0

    try:
        bmr = calculate_bmr(
            birth_year=int(birth_year),
            sex=str(sex) if sex else "other",
            weight_kg=float(weight_kg) if weight_kg is not None and float(weight_kg) > 0 else None,
            height_cm=float(height_cm) if height_cm is not None and float(height_cm) > 0 else None,
        )
    except Exception:
        return 2000.0, 75.0, 250.0, 65.0

    act_str = str(activity_level).strip().lower() if activity_level else "sedentary"
    act_multiplier = ACTIVITY_MULTIPLIERS.get(act_str, 1.2)

    goal_str = str(goal).strip().lower() if goal else "maintain"
    goal_delta = GOAL_ADJUSTMENTS.get(goal_str, 0.0)

    if target_kcal_override is not None and float(target_kcal_override) > 0:
        raw_target = float(target_kcal_override)
    else:
        tdee = bmr * act_multiplier
        raw_target = tdee + goal_delta

    target_kcal = clamp_goal_target(raw_target, bmr)

    if weight_kg is not None and float(weight_kg) > 0:
        target_protein = round(float(weight_kg) * 1.6, 1)
    else:
        target_protein = 75.0

    target_fat = round((target_kcal * 0.25) / 9.0, 1)
    target_carb = round(max(0.0, target_kcal - (target_protein * 4.0) - (target_fat * 9.0)) / 4.0, 1)

    return target_kcal, target_protein, target_carb, target_fat


def validate_non_shaming_text(text: str) -> bool:
    """Validate that copy contains no toxic or shaming language (Invariant I8).

    Bans terms such as 'over budget', 'you failed', 'streak lost', etc.
    Returns True if valid/safe, False if prohibited phrasing is detected.
    """
    if not text:
        return True
    lower_text = text.lower()
    for pattern in BANNED_SHAMING_PATTERNS:
        if re.search(pattern, lower_text):
            return False
    return True


def validate_weight_loss_rate(weight_kg: float, weekly_loss_target_kg: float) -> bool:
    """Validate that planned weight loss does not exceed 1% body weight / week (§11)."""
    if weight_kg <= 0 or weekly_loss_target_kg <= 0:
        return True
    return weekly_loss_target_kg <= (0.01 * weight_kg)
