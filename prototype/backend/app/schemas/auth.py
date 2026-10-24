"""Authentication request and response schemas."""

from datetime import datetime, timezone
import re
from typing import Optional
import uuid
from pydantic import BaseModel, ConfigDict, field_validator

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UserRegister(BaseModel):
    """User registration request enforcing Invariant I7 (age >= 18)."""

    email: str
    password: str
    birth_year: int
    sex: Optional[str] = "other"
    study_arm: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format.")
        return clean

    @field_validator("birth_year")
    @classmethod
    def validate_age_threshold(cls, v: int) -> int:
        current_year = datetime.now(timezone.utc).year
        if v > (current_year - 18):
            raise ValueError(
                f"Invariant I7 violated: Users must be at least 18 years old to register. Birth year {v} implies age < 18 in {current_year}."
            )
        if v < 1900:
            raise ValueError("Birth year must be 1900 or later.")
        return v

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> str:
        if v is None:
            return "other"
        clean = v.strip().lower()
        if clean not in ("male", "female", "other"):
            raise ValueError("Sex must be 'male', 'female', or 'other'.")
        return clean


class UserLogin(BaseModel):
    """User authentication login request."""

    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format.")
        return clean


class TokenResponse(BaseModel):
    """JWT bearer token response."""

    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)
