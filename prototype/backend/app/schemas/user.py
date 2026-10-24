"""User profile and study consent schemas."""

from datetime import datetime, timezone
from typing import Any, Optional
import uuid
from pydantic import BaseModel, ConfigDict, field_validator

from app.core.safety import DISCLAIMER_TEXT


class ProfileUpdate(BaseModel):
    """User profile update schema with safety and Invariant I7 validation."""

    birth_year: Optional[int] = None
    sex: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    activity_level: Optional[str] = None
    goal: Optional[str] = None
    target_kcal: Optional[float] = None
    diet_type: Optional[str] = None
    allergies: Optional[list[str]] = None
    conditions: Optional[list[str]] = None
    region: Optional[str] = None
    locale: Optional[str] = None
    cafeteria_id: Optional[str] = None

    @field_validator("birth_year")
    @classmethod
    def validate_birth_year(cls, v: Optional[int]) -> Optional[int]:
        if v is not None:
            current_year = datetime.now(timezone.utc).year
            if v > (current_year - 18):
                raise ValueError(
                    f"Invariant I7 violated: Users must be at least 18 years old. Birth year {v} implies age < 18 in {current_year}."
                )
            if v < 1900:
                raise ValueError("Birth year must be 1900 or later.")
        return v

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            clean = v.strip().lower()
            if clean not in ("male", "female", "other"):
                raise ValueError("Sex must be 'male', 'female', or 'other'.")
            return clean
        return v

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            clean = v.strip().lower()
            if clean not in ("maintain", "lose", "gain", "awareness"):
                raise ValueError("Goal must be 'maintain', 'lose', 'gain', or 'awareness'.")
            return clean
        return v

    @field_validator("diet_type")
    @classmethod
    def validate_diet_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            clean = v.strip().lower()
            if clean not in ("veg", "non_veg", "eggetarian", "vegan", "jain"):
                raise ValueError("Diet type must be 'veg', 'non_veg', 'eggetarian', 'vegan', or 'jain'.")
            return clean
        return v


class ProfileResponse(BaseModel):
    """User profile response including computed BMR and safe goal target (Invariant I8)."""

    user_id: uuid.UUID
    birth_year: int
    sex: str
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    activity_level: Optional[str] = None
    goal: Optional[str] = None
    target_kcal: Optional[float] = None
    clamped_target_kcal: Optional[float] = None
    bmr: float
    diet_type: Optional[str] = None
    allergies: list[str] = []
    conditions: list[str] = []
    region: Optional[str] = None
    locale: str = "en-IN"
    cafeteria_id: Optional[str] = None
    disclaimer: str = DISCLAIMER_TEXT

    model_config = ConfigDict(from_attributes=True)


class ConsentUpdate(BaseModel):
    """Unbundled, purpose-specific consent request (Invariant I6)."""

    purpose: str
    granted: bool
    notice_version: str = "v1.0"

    @field_validator("purpose")
    @classmethod
    def validate_purpose(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("core_logging", "model_improvement", "research_dataset"):
            raise ValueError(
                "Consent purpose must be one of 'core_logging', 'model_improvement', 'research_dataset'."
            )
        return clean


class ConsentResponse(BaseModel):
    """Granular consent record response."""

    id: uuid.UUID
    user_id: uuid.UUID
    purpose: str
    granted: bool
    notice_version: str
    granted_at: datetime
    revoked_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class TablewareSelectionItem(BaseModel):
    """Individual tableware archetype selection for Day 1 calibration (§6.2)."""

    unit: str  # e.g., 'small_bowl', 'standard_bowl', 'dinner_plate', 'cup_mug'
    size_label: str = "medium"  # 'small', 'medium', 'large'
    volume_ml: float = 350.0


class TablewareOnboardingRequest(BaseModel):
    """Day 1 Visual Kitchen Mirror setup request."""

    selections: list[TablewareSelectionItem] = []
    archetypes: Optional[list[TablewareSelectionItem]] = None
    context: str = "personal_home"

    @field_validator("selections", mode="before")
    @classmethod
    def populate_selections(cls, v: Any, info: Any) -> Any:
        return v

    def model_post_init(self, __context: Any) -> None:
        if not self.selections and self.archetypes:
            self.selections = self.archetypes
