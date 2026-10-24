"""Meal, MealItem, and ModelRun models."""

import uuid
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BigInt_compat, GUID, JSONB_compat

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.food import Food
    from app.models.question import Correction, Question


class Meal(Base):
    """Meal logging entity containing aggregate nutrient posteriors and context."""

    __tablename__ = "meals"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    meal_slot: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    image_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    image_purge_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    context: Mapped[str] = mapped_column(String(32), default="personal_home", nullable=False)
    kcal_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kcal_sd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kcal_p10: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kcal_p90: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_ask_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("source IN ('photo', 'voice', 'manual', 'text')", name="ck_meals_source"),
        CheckConstraint(
            "status IN ('pending', 'analyzed', 'failed', 'discarded')",
            name="ck_meals_status",
        ),
        CheckConstraint(
            "context IN ('personal_home', 'personal_office', 'commercial')",
            name="ck_meals_context",
        ),
        Index("ix_meals_user_local_date", "user_id", "local_date"),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="meals")
    items: Mapped[list["MealItem"]] = relationship(
        "MealItem",
        back_populates="meal",
        cascade="all, delete-orphan",
    )
    model_runs: Mapped[list["ModelRun"]] = relationship(
        "ModelRun",
        back_populates="meal",
        cascade="all, delete-orphan",
    )
    question: Mapped[Optional["Question"]] = relationship(
        "Question",
        back_populates="meal",
        uselist=False,
        cascade="all, delete-orphan",
    )


class MealItem(Base):
    """Discrete food component detected or logged in a meal."""

    __tablename__ = "meal_items"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    meal_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("meals.id", ondelete="CASCADE"),
        nullable=False,
    )
    food_id: Mapped[Optional[int]] = mapped_column(
        BigInt_compat,
        ForeignKey("foods.id", ondelete="SET NULL"),
        nullable=True,
    )
    dish_label_raw: Mapped[str] = mapped_column(String(255), nullable=False)
    resolution_tier: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    dish_uncertain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    unit: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    unit_count: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grams_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grams_sd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kcal_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kcal_sd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ask_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mu_log: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sigma2_log: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    portion_source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    vlm_grams_raw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_refill: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "resolution_tier IS NULL OR resolution_tier IN ('exact', 'fuzzy', 'decomposed', 'unresolved')",
            name="ck_meal_items_resolution_tier",
        ),
        CheckConstraint(
            "portion_source IS NULL OR portion_source IN ('model', 'prior', 'user')",
            name="ck_meal_items_portion_source",
        ),
    )

    # Relationships
    meal: Mapped["Meal"] = relationship("Meal", back_populates="items")
    food: Mapped[Optional["Food"]] = relationship("Food", back_populates="meal_items")
    questions: Mapped[list["Question"]] = relationship(
        "Question",
        back_populates="meal_item",
        cascade="all, delete-orphan",
    )
    corrections: Mapped[list["Correction"]] = relationship(
        "Correction",
        back_populates="meal_item",
        cascade="all, delete-orphan",
    )


class ModelRun(Base):
    """Inference execution metadata, prompt versions, and raw outputs (Invariant I10)."""

    __tablename__ = "model_runs"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    meal_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("meals.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    k_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_responses: Mapped[Any] = mapped_column(JSONB_compat, nullable=False)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    meal: Mapped["Meal"] = relationship("Meal", back_populates="model_runs")
