"""Question, Correction, and DailyBudget models."""

import uuid
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, GUID, JSONB_compat

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.meal import Meal, MealItem


class Question(Base):
    """Interactive question presented to user (Invariant I3: max 1 per meal)."""

    __tablename__ = "questions"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    meal_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("meals.id", ondelete="CASCADE"),
        nullable=False,
    )
    meal_item_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID,
        ForeignKey("meal_items.id", ondelete="CASCADE"),
        nullable=True,
    )
    question_type: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[Any] = mapped_column(JSONB_compat, nullable=False)
    ask_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_used: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    budget_remaining: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    asked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )
    answered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    answer: Mapped[Optional[Any]] = mapped_column(JSONB_compat, nullable=True)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        UniqueConstraint("meal_id", name="uq_questions_meal_id"),
        CheckConstraint(
            "question_type IN ('unit_count', 'unit_size', 'oil_level', 'dish_disambiguation')",
            name="ck_questions_question_type",
        ),
    )

    # Relationships
    meal: Mapped["Meal"] = relationship("Meal", back_populates="question")
    meal_item: Mapped[Optional["MealItem"]] = relationship("MealItem", back_populates="questions")


class Correction(Base):
    """User correction or calibration event (Invariant I5: strictly append-only)."""

    __tablename__ = "corrections"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    meal_item_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID,
        ForeignKey("meal_items.id", ondelete="CASCADE"),
        nullable=True,
    )
    field: Mapped[str] = mapped_column(String(32), nullable=False)
    old_value: Mapped[Optional[Any]] = mapped_column(JSONB_compat, nullable=True)
    new_value: Mapped[Any] = mapped_column(JSONB_compat, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "field IN ('dish', 'unit', 'unit_count', 'grams', 'oil_level')",
            name="ck_corrections_field",
        ),
        CheckConstraint(
            "source IN ('question', 'manual_edit', 'scale')",
            name="ck_corrections_source",
        ),
        Index("ix_corrections_user_created", "user_id", "created_at"),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="corrections")
    meal_item: Mapped[Optional["MealItem"]] = relationship("MealItem", back_populates="corrections")


class DailyBudget(Base):
    """Daily question budget tracker enforcing Invariant I3."""

    __tablename__ = "daily_budgets"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    local_date: Mapped[date] = mapped_column(Date, primary_key=True)
    budget_total: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    budget_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    __table_args__ = (
        CheckConstraint("budget_used <= budget_total", name="ck_daily_budgets_budget_limit"),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="daily_budgets")
