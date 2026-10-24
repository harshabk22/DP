"""Consent management endpoints implementing Invariant I6 (unbundled consent)."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.user import Consent, User
from app.schemas.user import ConsentResponse, ConsentUpdate

router = APIRouter(prefix="/me/consents", tags=["consents"])


@router.get(
    "",
    response_model=list[ConsentResponse],
    summary="Get all unbundled consent records for the authenticated user (Invariant I6)",
)
def get_consents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ConsentResponse]:
    """Retrieve history of granular research and logging consents."""
    consents = (
        db.query(Consent)
        .filter(Consent.user_id == current_user.id)
        .order_by(Consent.granted_at.desc())
        .all()
    )
    return [ConsentResponse.model_validate(c) for c in consents]


@router.post(
    "",
    response_model=ConsentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record or update a single purpose-specific consent (Invariant I6)",
)
def record_consent(
    consent_in: ConsentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConsentResponse:
    """Record an unbundled consent grant or revocation."""
    now = datetime.now(timezone.utc)
    revoked_at = None if consent_in.granted else now

    consent = Consent(
        user_id=current_user.id,
        purpose=consent_in.purpose,
        granted=consent_in.granted,
        notice_version=consent_in.notice_version,
        granted_at=now,
        revoked_at=revoked_at,
    )
    db.add(consent)
    db.commit()
    db.refresh(consent)

    return ConsentResponse.model_validate(consent)
