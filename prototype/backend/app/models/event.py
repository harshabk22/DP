"""Event model for telemetry, UX engagement, and research analysis."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BigInt_compat, GUID, JSONB_compat

if TYPE_CHECKING:
    from app.models.user import User


class Event(Base):
    """Telemetry and interaction event stream record."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInt_compat, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
    )
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB_compat, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_events_user_type_created", "user_id", "type", "created_at"),
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="events")
