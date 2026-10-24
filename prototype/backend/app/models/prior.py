"""UnitPrior and UnitPriorHistory models for active Bayesian tableware calibration."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BigInt_compat, GUID

if TYPE_CHECKING:
    from app.models.user import User


class UnitPrior(Base):
    """Personalized posterior distribution for tableware unit scale (theta_u,m)."""

    __tablename__ = "unit_priors"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    unit: Mapped[str] = mapped_column(String(64), primary_key=True)
    context: Mapped[str] = mapped_column(String(32), primary_key=True, default="personal_home")
    mu_log: Mapped[float] = mapped_column(Float, nullable=False)
    sigma2_log: Mapped[float] = mapped_column(Float, nullable=False)
    n_obs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    consecutive_rejects: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_reject_sign: Mapped[int] = mapped_column(SmallInteger, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "context IN ('personal_home', 'personal_office', 'commercial')",
            name="ck_unit_priors_context",
        ),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="unit_priors")


class UnitPriorHistory(Base):
    """Posterior trajectory log for Bayesian update convergence analysis (Figure 2a)."""

    __tablename__ = "unit_prior_history"

    id: Mapped[int] = mapped_column(BigInt_compat, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    unit: Mapped[str] = mapped_column(String(64), nullable=False)
    context: Mapped[str] = mapped_column(String(32), default="personal_home", nullable=False)
    mu_log: Mapped[float] = mapped_column(Float, nullable=False)
    sigma2_log: Mapped[float] = mapped_column(Float, nullable=False)
    n_obs: Mapped[int] = mapped_column(Integer, nullable=False)
    triggered_by: Mapped[Optional[uuid.UUID]] = mapped_column(GUID, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="unit_prior_history")
