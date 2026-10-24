"""Question and answering schemas enforcing Invariants I3 and I4."""

from datetime import datetime
from typing import Any, Optional
import uuid
from pydantic import BaseModel, ConfigDict, field_validator


class QuestionOption(BaseModel):
    """Discrete tap-answerable choice (Invariant I4: max 4 options per question)."""

    label: str
    value: Any

    model_config = ConfigDict(from_attributes=True)


class QuestionAnswerRequest(BaseModel):
    """Request payload to answer or skip an allocated meal question (Invariants I4, I5)."""

    answer: Optional[Any] = None
    skipped: bool = False

    @field_validator("answer")
    @classmethod
    def validate_tap_answer(cls, v: Optional[Any]) -> Optional[Any]:
        # Free-text paragraphs are strictly prohibited by Invariant I4
        if isinstance(v, str) and len(v.strip()) > 64:
            raise ValueError(
                "Invariant I4 violated: Questions are tap-only with discrete choices. Free-form text responses are prohibited."
            )
        return v


class QuestionResponse(BaseModel):
    """Response representation of an allocated question record."""

    id: uuid.UUID
    meal_id: uuid.UUID
    meal_item_id: Optional[uuid.UUID] = None
    question_type: str
    prompt_text: str
    options: list[Any]
    ask_value: float
    budget_remaining: int
    policy_name: str
    asked_at: datetime
    answered_at: Optional[datetime] = None
    answer: Optional[Any] = None
    skipped: bool = False

    model_config = ConfigDict(from_attributes=True)
