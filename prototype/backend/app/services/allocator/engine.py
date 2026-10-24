"""Thali-Robust Value-of-Information (VoI) Budget Allocator.

Adheres to MESSMATE_SPEC.md §6.3 and Master Blueprint §3.7:
- Goodman's Law of Total Variance across mixture labels and portion mass variance:
    Var(kcal) = E[X]^2 * Var(Y) + E[Y]^2 * Var(X) + Var(X) * Var(Y)
    where X = energy density (kcal/g) and Y = mass (grams).
- Strict 25 kcal Chutney Theorem cutoff (V_FLOOR = 25.0 kcal):
    If v_best < V_FLOOR, the assistant stays silent.
    Bounds total daily unasked uncertainty to < 75 kcal/day (sigma < 43.3 kcal/day),
    submerging it beneath physiological metabolic noise.
- Thali-robust ask value: v_i = sd_kcal_i (eliminates multi-dish Pythagorean suppression).
- Budget-pressure policy:
    P = B / meals_remaining.
    If P >= 1.0, ask top item directly (if v_best >= V_FLOOR).
    If P < 1.0, ask only if v_best >= max(V_FLOOR, quantile(v_history, 1.0 - P)).
- Invariant I3: maximum daily question budget enforced atomically via DB locks.
- Invariant I4: tap-only questions with at most 4 discrete choices.
"""

from datetime import date
import math
from typing import Any, Optional, Sequence
import uuid
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.database import is_postgres
from app.models.question import DailyBudget

# Algorithmic constants (§3.7 Master Blueprint)
V_FLOOR: float = 25.0  # Strict Chutney Theorem cutoff in kcal


class CandidateLabel(BaseModel):
    """Candidate dish classification under VLM label entropy."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., description="Food/dish candidate name")
    probability: float = Field(..., ge=0.0, le=1.0, description="Mixture probability pi_c")
    kcal_per_100g: float = Field(..., ge=0.0, description="Energy density per 100g")
    sigma_kcal_per_100g: float = Field(default=0.0, ge=0.0, description="Internal uncertainty in energy density")


class GeneratedQuestion(BaseModel):
    """Candidate interactive clarification question meeting Invariants I3 and I4."""

    model_config = ConfigDict(from_attributes=True)

    meal_id: uuid.UUID = Field(..., description="Target meal ID")
    meal_item_id: Optional[uuid.UUID] = Field(default=None, description="Specific item under question")
    question_type: str = Field(..., description="Type of question: unit_count, dish_disambiguation, unit_size, oil_level")
    prompt_text: str = Field(..., description="Human-readable prompt text")
    options: list[str] = Field(..., description="Tap-only choices (length <= 4 per Invariant I4)")
    ask_value: float = Field(..., description="Expected calorie uncertainty reduction v_i in kcal")
    threshold_used: float = Field(..., description="Threshold applied by budget-pressure policy")
    budget_remaining: int = Field(..., description="Budget B prior to question deduction")
    policy_name: str = Field(default="budget_pressure_voi", description="Allocation policy identifier")


class ItemCandidate(BaseModel):
    """Normalized candidate item for VoI question evaluation."""

    model_config = ConfigDict(from_attributes=True)

    meal_item_id: Optional[uuid.UUID] = None
    food_name: str = "dish"
    sd_kcal: float = 0.0
    unit: Optional[str] = None
    unit_count: Optional[float] = 1.0
    askable: bool = True
    question_type: str = "unit_count"
    candidate_options: Optional[list[str]] = None
    grams_mean: Optional[float] = None
    grams_sd: Optional[float] = None
    candidate_labels: Optional[list[CandidateLabel]] = None


def compute_goodman_variance(
    mean_x: float,
    var_x: float,
    mean_y: float,
    var_y: float,
) -> float:
    """Exact Goodman (1960) product variance for independent random variables X and Y (§3.7).

    Var(X * Y) = E[X]^2 * Var(Y) + E[Y]^2 * Var(X) + Var(X) * Var(Y)

    Args:
        mean_x: Expected energy density E[X] in kcal/g.
        var_x: Variance of energy density Var(X) in (kcal/g)^2 across label mixture.
        mean_y: Expected mass E[Y] in grams.
        var_y: Variance of mass Var(Y) in grams^2.

    Returns:
        Total product variance Var(kcal) in kcal^2.
    """
    return (mean_x ** 2) * var_y + (mean_y ** 2) * var_x + (var_x * var_y)


def compute_label_mixture_moments(
    labels: Sequence[CandidateLabel | dict[str, Any]],
) -> tuple[float, float]:
    """Compute energy density moments (E[X], Var(X)) in kcal/g under label entropy (§3.7).

    By the Law of Total Expectation:
        E[X] = sum_c pi_c * mu_{X, c}
    By the Law of Total Variance (Eve's Law):
        Var(X) = sum_c pi_c * sigma^2_{X, c} + sum_c pi_c * (mu_{X, c} - E[X])^2

    Returns:
        (E[X], Var(X)) in kcal/g and (kcal/g)^2.
    """
    if not labels:
        return 0.0, 0.0

    parsed: list[tuple[float, float, float]] = []
    total_w = 0.0

    for item in labels:
        if isinstance(item, CandidateLabel):
            prob = float(item.probability)
            k100 = float(item.kcal_per_100g)
            s100 = float(item.sigma_kcal_per_100g)
        elif isinstance(item, dict):
            prob = float(item.get("probability", item.get("prob", item.get("weight", 1.0))))
            k100 = float(item.get("kcal_per_100g", item.get("kcal", 0.0)))
            s100 = float(item.get("sigma_kcal_per_100g", 0.0))
        else:
            continue

        mu_g = k100 / 100.0
        var_internal = (s100 / 100.0) ** 2
        parsed.append((prob, mu_g, var_internal))
        total_w += prob

    if not parsed or total_w <= 0.0:
        return 0.0, 0.0

    # Normalize mixture probabilities
    normed = [(p / total_w, mu, var) for p, mu, var in parsed]

    # E[X] = sum_c pi_c * mu_{X, c}
    e_x = sum(p * mu for p, mu, _ in normed)

    # Var(X) = sum_c pi_c * sigma^2_{X, c} + sum_c pi_c * (mu_{X, c} - E[X])^2
    var_x = sum(p * var for p, _, var in normed) + sum(p * ((mu - e_x) ** 2) for p, mu, _ in normed)

    return e_x, var_x


def compute_goodman_item_uncertainty(
    grams_mean: float,
    grams_sd: float,
    labels: Sequence[CandidateLabel | dict[str, Any]],
) -> float:
    """Calculate exact calorie uncertainty (sd_kcal) via Goodman's product formula (§3.7).

    Combines portion mass distribution moments with dish label mixture entropy.

    Args:
        grams_mean: Expected mass E[Y] in grams.
        grams_sd: Mass standard deviation in grams.
        labels: Candidate dish classifications with probabilities and energy densities.

    Returns:
        Exact calorie standard deviation sd_kcal in kcal.
    """
    mean_y = max(float(grams_mean), 0.0)
    var_y = max(float(grams_sd), 0.0) ** 2

    mean_x, var_x = compute_label_mixture_moments(labels)

    var_kcal = compute_goodman_variance(mean_x, var_x, mean_y, var_y)
    return math.sqrt(max(var_kcal, 0.0))


def _get_item_attr(item: Any, attr: str, default: Any = None) -> Any:
    """Helper to safely extract field from dict, Pydantic model, or ORM object."""
    if isinstance(item, dict):
        return item.get(attr, default)
    return getattr(item, attr, default)


def _get_sd_kcal(item: Any) -> float:
    """Extract item calorie uncertainty v_i = sd_kcal.

    If direct sd_kcal is zero or missing, dynamically evaluates Goodman's
    exact product formula under label mixture entropy if mass moments and
    candidate labels are present.
    """
    for field in ("sd_kcal", "ask_value", "kcal_sd"):
        val = _get_item_attr(item, field, None)
        if val is not None:
            try:
                fval = float(val)
                if fval > 0.0:
                    return fval
            except (ValueError, TypeError):
                continue

    grams_mean = _get_item_attr(item, "grams_mean", None)
    grams_sd = _get_item_attr(item, "grams_sd", None)
    candidate_labels = _get_item_attr(item, "candidate_labels", None)
    if grams_mean is not None and grams_sd is not None and candidate_labels:
        try:
            return compute_goodman_item_uncertainty(float(grams_mean), float(grams_sd), candidate_labels)
        except Exception:
            pass

    return 0.0


def build_tap_question(
    question_type: str,
    food_name: str,
    unit: Optional[str] = None,
    candidate_options: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    """Build prompt text and at most 4 tap options adhering to Invariant I4."""
    if question_type == "dish_disambiguation" and candidate_options:
        prompt = f"Which dish was this?"
        options = list(candidate_options[:4])
        return prompt, options

    if question_type == "unit_size":
        unit_label = unit or "portion"
        prompt = f"What size was your {unit_label} of {food_name}?"
        options = ["Small", "Medium", "Large"]
        return prompt, options

    if question_type == "oil_level":
        prompt = f"How much oil or ghee was added to {food_name}?"
        options = ["Light", "Normal", "Extra"]
        return prompt, options

    # Default: unit_count question
    if unit and unit not in ("portion", "serving", "unknown"):
        clean_unit = unit.replace("_", " ")
        prompt = f"How many {clean_unit}s of {food_name}?"
    else:
        prompt = f"How many servings of {food_name}?"

    options = ["1", "2", "3", "4+"]
    return prompt, options


def evaluate_budget_pressure_ask(
    meal_id: uuid.UUID,
    items: Sequence[Any],
    budget_remaining: int,
    meals_remaining: int = 1,
    v_history: Optional[Sequence[float]] = None,
    v_floor: float = V_FLOOR,
    policy_name: str = "budget_pressure_voi",
) -> Optional[GeneratedQuestion]:
    """Thali-robust online VoI question selection under daily budget pressure (§6.3 & §3.7).

    Enforces the Chutney Theorem:
    1. B <= 0 -> None
    2. Candidate ask value v_i = sd_kcal_i (via direct uncertainty or Goodman's product formula)
    3. Best candidate v_best = max_i v_i
    4. If v_best < V_FLOOR (25.0 kcal) -> None (never interrupt for trivial gains)
    5. Budget pressure P = B / max(meals_remaining, 1)
    6. If P >= 1.0 -> ask top item immediately (threshold = V_FLOOR)
    7. If P < 1.0 -> ask only if v_best >= max(V_FLOOR, quantile(v_history, 1.0 - P))

    Returns:
        GeneratedQuestion if criteria satisfied, else None.
    """
    b = int(budget_remaining)
    if b <= 0:
        return None

    # Filter askable candidates with positive calorie uncertainty
    candidates = [
        item for item in items
        if _get_item_attr(item, "askable", True)
        and _get_item_attr(item, "portion_source", "") != "user"
        and not _get_item_attr(item, "is_refill", False)
        and _get_sd_kcal(item) > 0.0
    ]
    if not candidates:
        return None

    # Find highest uncertainty item (Thali-robust: no multi-dish suppression)
    best_item = max(candidates, key=_get_sd_kcal)
    v_best = _get_sd_kcal(best_item)

    # Chutney Theorem cutoff: strictly suppress asking if below 25.0 kcal floor
    if v_best < float(v_floor):
        return None

    # Calculate budget pressure
    meals_rem = max(int(meals_remaining), 1)
    pressure = float(b) / float(meals_rem)

    if pressure >= 1.0:
        # Abundant budget: ask top item directly
        threshold_used = float(v_floor)
    else:
        # Budget is tight (P < 1.0): apply historical quantile threshold
        clean_history = [float(v) for v in (v_history or []) if math.isfinite(float(v))]
        if len(clean_history) > 0:
            q_level = max(0.0, min(1.0, 1.0 - pressure))
            hist_q = float(np.quantile(clean_history, q_level))
            threshold_used = max(float(v_floor), hist_q) if math.isfinite(hist_q) else max(float(v_floor), 35.0)
        else:
            threshold_used = max(float(v_floor), 35.0)

        if v_best < threshold_used:
            return None

    # Construct question details
    q_type = _get_item_attr(best_item, "question_type", "unit_count")
    meal_item_id = _get_item_attr(best_item, "meal_item_id", _get_item_attr(best_item, "id", None))
    food_name = _get_item_attr(
        best_item,
        "dish_label_raw",
        _get_item_attr(best_item, "food_name", _get_item_attr(best_item, "name", "dish")),
    )
    unit = _get_item_attr(best_item, "unit", None)
    candidate_options = _get_item_attr(best_item, "candidate_options", None)

    # Fallback options from candidate_labels if dish_disambiguation lacks explicit candidate_options
    if not candidate_options and q_type == "dish_disambiguation":
        cand_labels = _get_item_attr(best_item, "candidate_labels", None)
        if cand_labels:
            candidate_options = [
                l.name if isinstance(l, CandidateLabel) else str(l.get("name", "dish"))
                for l in cand_labels
            ]

    prompt_text, options = build_tap_question(q_type, food_name, unit, candidate_options)

    # Invariant I4: Strictly clamp to at most 4 options
    options = options[:4]

    return GeneratedQuestion(
        meal_id=meal_id,
        meal_item_id=meal_item_id,
        question_type=q_type,
        prompt_text=prompt_text,
        options=options,
        ask_value=v_best,
        threshold_used=threshold_used,
        budget_remaining=b,
        policy_name=policy_name,
    )


def reserve_question_slot(
    session: Session,
    user_id: uuid.UUID,
    today: date,
    budget_total: int = 3,
) -> tuple[bool, int]:
    """Atomically check and reserve a daily question slot from DailyBudget.

    Enforces Invariant I3: maximum daily question budget.
    Executes an atomic conditional UPDATE query safe across both SQLite and PostgreSQL:
      UPDATE daily_budgets
      SET budget_used = budget_used + 1
      WHERE user_id = :user_id
        AND local_date = :today
        AND budget_used < budget_total;

    Returns:
        (success, remaining_slots_after_reservation)
        If exhausted, returns (False, 0).
    """
    # Ensure the row exists for the user and day
    existing = session.scalar(
        select(DailyBudget).where(
            DailyBudget.user_id == user_id,
            DailyBudget.local_date == today,
        )
    )
    if existing is None:
        try:
            with session.begin_nested():
                budget = DailyBudget(
                    user_id=user_id,
                    local_date=today,
                    budget_total=budget_total,
                    budget_used=0,
                )
                session.add(budget)
                session.flush()
        except IntegrityError:
            pass

    # Execute conditional atomic update safe across both SQLite and PostgreSQL
    stmt = (
        update(DailyBudget)
        .where(
            DailyBudget.user_id == user_id,
            DailyBudget.local_date == today,
            DailyBudget.budget_used < DailyBudget.budget_total,
        )
        .values(budget_used=DailyBudget.budget_used + 1)
    )
    res = session.execute(stmt)
    session.flush()

    if res.rowcount > 0:
        budget = session.scalar(
            select(DailyBudget).where(
                DailyBudget.user_id == user_id,
                DailyBudget.local_date == today,
            )
        )
        if budget is not None:
            session.refresh(budget)
            remaining = max(0, budget.budget_total - budget.budget_used)
        else:
            remaining = 0
        return True, remaining
    else:
        return False, 0


def get_budget_remaining(
    session: Session,
    user_id: uuid.UUID,
    today: date,
    default_total: int = 3,
) -> int:
    """Query current remaining questions for the given user and day."""
    stmt = select(DailyBudget).where(
        DailyBudget.user_id == user_id,
        DailyBudget.local_date == today,
    )
    budget = session.scalar(stmt)
    if budget is None:
        return default_total
    return max(0, budget.budget_total - budget.budget_used)
