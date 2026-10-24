"""Daily rollup and weekly digest summary schemas."""

from datetime import date
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict

from app.core.safety import DISCLAIMER_TEXT
from app.schemas.meal import BudgetDetail, MealDetailResponse


class FoodSwapResponse(BaseModel):
    """Actionable, high-density food swap proposal (§10)."""

    food_id: int
    name: str
    food_group: Optional[str] = None
    target_nutrient: str
    nutrient_density: float
    serving_description: str
    rationale: str

    model_config = ConfigDict(from_attributes=True)


class DayRollupResponse(BaseModel):
    """Daily nutrition rollup and question budget state."""

    date: date
    meal_count: int
    meals: list[MealDetailResponse] = []
    total_kcal_mean: float = 0.0
    total_kcal_sd: float = 0.0
    total_kcal_p10: float = 0.0
    total_kcal_p90: float = 0.0
    total_protein_g: float = 0.0
    total_fat_g: float = 0.0
    total_carb_g: float = 0.0
    total_fibre_g: float = 0.0
    target_kcal: Optional[float] = None
    target_protein_g: Optional[float] = None
    target_carb_g: Optional[float] = None
    target_fat_g: Optional[float] = None
    remaining_budget_kcal: Optional[float] = None
    budget: BudgetDetail
    dinner_advice: Optional[str] = None
    dinner_swaps: list[FoodSwapResponse] = []
    disclaimer: str = DISCLAIMER_TEXT

    model_config = ConfigDict(from_attributes=True)


class WeekSummaryResponse(BaseModel):
    """Weekly digest summary adhering to Invariant I8 (no streak/loss framing)."""

    iso_week: str
    days_logged: int
    mean_daily_kcal: float = 0.0
    mean_daily_protein_g: float = 0.0
    mean_daily_fibre_g: float = 0.0
    protein_gap_g: float = 0.0
    fibre_gap_g: float = 0.0
    swaps: list[FoodSwapResponse] = []
    disclaimer: str = DISCLAIMER_TEXT

    model_config = ConfigDict(from_attributes=True)
