"""Schemas for Conversational Nutrition Co-Pilot."""

from datetime import date, datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatRequest(BaseModel):
    """User message sent to the Nutrition Co-Pilot."""

    message: str = Field(..., min_length=1, description="User question or statement")
    prompt: Optional[str] = Field(None, description="Optional alias for message")
    date_query: Optional[date] = Field(None, description="Optional target date to query; defaults to today")

    @model_validator(mode="before")
    @classmethod
    def resolve_message_or_prompt(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            msg = data.get("message")
            prompt = data.get("prompt")
            if (msg is None or msg == "") and prompt is not None:
                data["message"] = prompt
            elif (prompt is None or prompt == "") and msg is not None:
                data["prompt"] = msg
        return data


class MacroOverview(BaseModel):
    """Macro breakdown for context."""

    target_kcal: float
    consumed_kcal: float
    remaining_kcal: float
    protein_g: float
    carb_g: float
    fat_g: float


class ChatResponse(BaseModel):
    """Grounded Co-Pilot response."""

    reply: str
    macros: MacroOverview


class ChatMessageResponse(BaseModel):
    """Persisted chat message model."""

    id: Any
    user_id: Any
    role: str
    message: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
