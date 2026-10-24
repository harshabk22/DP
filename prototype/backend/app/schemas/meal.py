"""Meal ingestion, detailed response, and item update schemas."""

from datetime import datetime
from typing import Any, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field


class MealUploadResponse(BaseModel):
    """Asynchronous 202 Accepted response for meal uploads."""

    meal_id: uuid.UUID
    status: str = "pending"
    message: str = "Meal received and queued for analysis."

    model_config = ConfigDict(from_attributes=True)


class MealTotalDetail(BaseModel):
    """Aggregate distribution and macronutrient summary for a meal (Invariants I1, I2)."""

    kcal_mean: float
    kcal_sd: float
    kcal_p10: float
    kcal_p90: float
    protein_g: Optional[float] = 0.0
    fat_g: Optional[float] = 0.0
    carb_g: Optional[float] = 0.0
    fibre_g: Optional[float] = 0.0

    model_config = ConfigDict(from_attributes=True)


class MealItemDetail(BaseModel):
    """Individual item resolution, portion estimation, and ask-value details."""

    id: uuid.UUID
    dish_label: str
    food_id: Optional[int] = None
    resolution_tier: Optional[str] = None
    unit: Optional[str] = None
    unit_count: Optional[float] = None
    grams_mean: Optional[float] = None
    grams_sd: Optional[float] = None
    kcal_mean: Optional[float] = None
    kcal_sd: Optional[float] = None
    portion_source: Optional[str] = None
    ask_value: Optional[float] = None
    is_refill: bool = False
    vlm_grams_raw: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class QuestionOptionDetail(BaseModel):
    """Discrete tap-answerable option for push notification or in-app card (Invariant I4)."""

    label: str
    value: Any


class QuestionEmbedded(BaseModel):
    """Interactive question presented to user if allocator triggers (Invariants I3, I4, I9)."""

    id: uuid.UUID
    type: str
    prompt: str
    options: list[QuestionOptionDetail]

    model_config = ConfigDict(from_attributes=True)


class BudgetDetail(BaseModel):
    """Daily question budget tracker state."""

    total: int = 3
    used: int = 0
    remaining: Optional[int] = None


class MealDetailResponse(BaseModel):
    """Complete meal analysis output according to MESSMATE_SPEC.md §8."""

    meal_id: uuid.UUID
    status: str
    captured_at: datetime
    context: str = "personal_home"
    total: Optional[MealTotalDetail] = None
    items: list[MealItemDetail] = []
    question: Optional[QuestionEmbedded] = None
    budget: BudgetDetail

    model_config = ConfigDict(from_attributes=True)


class MealItemUpdate(BaseModel):
    """Manual adjustment or refill chip update schema (appends to corrections I5)."""

    unit_count: Optional[float] = None
    unit: Optional[str] = None
    dish_label: Optional[str] = None
    grams: Optional[float] = None
    oil_level: Optional[str] = None
    is_refill: Optional[bool] = None


class MealTextRequest(BaseModel):
    """Natural language text or voice transcript quick-log request."""

    text: str = Field(..., min_length=2, description="Natural language description of food consumed")
    meal_slot: Optional[str] = "lunch"
    context: Optional[str] = "personal_home"
    source: Optional[str] = "text"
    captured_at: Optional[str] = None

