"""Food and CafeteriaMenu reference models."""

from typing import TYPE_CHECKING, Any, Optional
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Array_compat, Base, BigInt_compat, JSONB_compat

if TYPE_CHECKING:
    from app.models.meal import MealItem


class Food(Base):
    """Certified reference food and nutrient composition record."""

    __tablename__ = "foods"

    id: Mapped[int] = mapped_column(BigInt_compat, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    name_en: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    name_local: Mapped[dict] = mapped_column(JSONB_compat, default=dict, nullable=False)
    aliases: Mapped[list[str]] = mapped_column(Array_compat(String), default=list, nullable=False)
    food_group: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    density_g_per_ml: Mapped[float] = mapped_column(Float, default=1.00, nullable=False)
    kcal_per_100g: Mapped[float] = mapped_column(Float, nullable=False)
    protein_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fat_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    carb_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fibre_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    micros: Mapped[dict] = mapped_column(JSONB_compat, default=dict, nullable=False)
    default_unit: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    pop_grams_per_unit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    embedding_json: Mapped[Optional[Any]] = mapped_column(JSONB_compat, nullable=True)
    provenance_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "source IN ('FNDDS', 'INDB', 'IFCT2017', 'CURATED')",
            name="ck_foods_source",
        ),
        UniqueConstraint("source", "source_code", name="uq_foods_source_code"),
    )

    # Relationships
    cafeteria_menus: Mapped[list["CafeteriaMenu"]] = relationship(
        "CafeteriaMenu",
        back_populates="food",
    )
    meal_items: Mapped[list["MealItem"]] = relationship(
        "MealItem",
        back_populates="food",
    )

    @property
    def embedding(self) -> Optional[Any]:
        """Convenience property accessing the embedding representation."""
        return self.embedding_json

    @embedding.setter
    def embedding(self, value: Any) -> None:
        self.embedding_json = value


class CafeteriaMenu(Base):
    """Institutional or cafeteria menu item mapping."""

    __tablename__ = "cafeteria_menus"

    id: Mapped[int] = mapped_column(BigInt_compat, primary_key=True, autoincrement=True)
    cafeteria_id: Mapped[str] = mapped_column(String(64), default="campus_default", nullable=False)
    day_of_week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    meal_slot: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    food_id: Mapped[Optional[int]] = mapped_column(
        BigInt_compat,
        ForeignKey("foods.id", ondelete="SET NULL"),
        nullable=True,
    )
    standard_serving_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        CheckConstraint("day_of_week IS NULL OR day_of_week BETWEEN 0 AND 6", name="ck_cafeteria_menus_day_of_week"),
        CheckConstraint(
            "meal_slot IS NULL OR meal_slot IN ('breakfast', 'lunch', 'snack', 'dinner')",
            name="ck_cafeteria_menus_meal_slot",
        ),
    )

    # Relationships
    food: Mapped[Optional["Food"]] = relationship("Food", back_populates="cafeteria_menus")
