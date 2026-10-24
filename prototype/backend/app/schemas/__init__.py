"""Pydantic schemas for MessMate API."""

from app.schemas.auth import TokenResponse, UserLogin, UserRegister
from app.schemas.user import (
    ConsentResponse,
    ConsentUpdate,
    ProfileResponse,
    ProfileUpdate,
)
from app.schemas.meal import (
    MealDetailResponse,
    MealItemDetail,
    MealItemUpdate,
    MealTotalDetail,
    MealUploadResponse,
)
from app.schemas.question import (
    QuestionAnswerRequest,
    QuestionOption,
    QuestionResponse,
)
from app.schemas.rollup import (
    DayRollupResponse,
    FoodSwapResponse,
    WeekSummaryResponse,
)

__all__ = [
    "UserRegister",
    "UserLogin",
    "TokenResponse",
    "ProfileUpdate",
    "ProfileResponse",
    "ConsentUpdate",
    "ConsentResponse",
    "MealUploadResponse",
    "MealDetailResponse",
    "MealItemDetail",
    "MealTotalDetail",
    "MealItemUpdate",
    "QuestionAnswerRequest",
    "QuestionOption",
    "QuestionResponse",
    "DayRollupResponse",
    "WeekSummaryResponse",
    "FoodSwapResponse",
]
