"""User, Profile, and Consent models."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Array_compat, Base, GUID

if TYPE_CHECKING:
    from app.models.chat import ChatMessage
    from app.models.meal import Meal
    from app.models.question import Correction, DailyBudget
    from app.models.prior import UnitPrior, UnitPriorHistory
    from app.models.event import Event


class User(Base):
    """User account model."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    study_arm: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "study_arm IS NULL OR study_arm IN ('A_fixed', 'B_adaptive')",
            name="ck_users_study_arm",
        ),
    )

    # Relationships
    profile: Mapped[Optional["Profile"]] = relationship(
        "Profile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    consents: Mapped[list["Consent"]] = relationship(
        "Consent",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    meals: Mapped[list["Meal"]] = relationship(
        "Meal",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    corrections: Mapped[list["Correction"]] = relationship(
        "Correction",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    daily_budgets: Mapped[list["DailyBudget"]] = relationship(
        "DailyBudget",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    unit_priors: Mapped[list["UnitPrior"]] = relationship(
        "UnitPrior",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    unit_prior_history: Mapped[list["UnitPriorHistory"]] = relationship(
        "UnitPriorHistory",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    events: Mapped[list["Event"]] = relationship(
        "Event",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Profile(Base):
    """User profile information, physical metrics, and dietary preferences."""

    __tablename__ = "profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    birth_year: Mapped[int] = mapped_column(Integer, nullable=False)
    sex: Mapped[str] = mapped_column(String(16), nullable=False)
    height_cm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weight_kg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    activity_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    goal: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    diet_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    allergies: Mapped[list[str]] = mapped_column(Array_compat(String), default=list, nullable=False)
    conditions: Mapped[list[str]] = mapped_column(Array_compat(String), default=list, nullable=False)
    region: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    locale: Mapped[str] = mapped_column(String(16), default="en-IN", nullable=False)
    cafeteria_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("birth_year BETWEEN 1900 AND 2100", name="ck_profiles_birth_year"),
        CheckConstraint("sex IN ('male', 'female', 'other')", name="ck_profiles_sex"),
        CheckConstraint(
            "activity_level IS NULL OR activity_level IN ('sedentary', 'light', 'moderate', 'active', 'very_active')",
            name="ck_profiles_activity_level",
        ),
        CheckConstraint(
            "goal IS NULL OR goal IN ('maintain', 'lose', 'gain', 'awareness')",
            name="ck_profiles_goal",
        ),
        CheckConstraint(
            "diet_type IS NULL OR diet_type IN ('veg', 'non_veg', 'eggetarian', 'vegan', 'jain')",
            name="ck_profiles_diet_type",
        ),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="profile")


class Consent(Base):
    """Granular user consent record (Invariant I6: unbundled, per-purpose)."""

    __tablename__ = "consents"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    granted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    notice_version: Mapped[str] = mapped_column(String(64), nullable=False)
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "purpose IN ('core_logging', 'model_improvement', 'research_dataset')",
            name="ck_consents_purpose",
        ),
        Index("ix_consents_user_purpose_granted", "user_id", "purpose", "granted_at"),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="consents")
