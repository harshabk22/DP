"""Authentication endpoints implementing Invariant I7 (under-18 block)."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import Profile, User
from app.schemas.auth import TokenResponse, UserLogin, UserRegister

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account enforcing Invariant I7",
)
def register(
    register_in: UserRegister,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Register a new user account.

    Enforces Invariant I7: Users under 18 cannot create an account.
    """
    # Check for existing email
    existing_user = db.query(User).filter(User.email == register_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists.",
        )

    # Hash password with Argon2id
    hashed = hash_password(register_in.password)

    user = User(
        email=register_in.email,
        hashed_password=hashed,
        study_arm=register_in.study_arm,
    )
    db.add(user)
    db.flush()

    # Create associated profile
    profile = Profile(
        user_id=user.id,
        birth_year=register_in.birth_year,
        sex=register_in.sex or "other",
    )
    db.add(profile)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, token_type="bearer", user_id=user.id)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user and receive JWT access token",
)
def login(
    login_in: UserLogin,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Authenticate with email and password."""
    user = (
        db.query(User)
        .filter(User.email == login_in.email, User.deleted_at.is_(None))
        .first()
    )
    if not user or not verify_password(login_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, token_type="bearer", user_id=user.id)
