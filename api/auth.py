"""
api/auth.py
───────────
JWT authentication endpoints: register and login.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.context import CryptContext

from database import User
from api.schemas import UserCreate, UserLogin, TokenResponse, UserOut
from api.deps import get_db, create_access_token

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new farmer account",
    description=(
        "Creates a new user with a phone number and password. "
        "The phone number must be unique. Passwords are stored as bcrypt hashes."
    ),
)
async def register(body: UserCreate, db: AsyncSession = Depends(get_db)):
    """
    **Register** a new farmer account.

    - **phone**: 10–15 digit mobile number (used as username)
    - **password**: minimum 6 characters
    - **full_name**: optional display name

    Returns the created user profile.
    """
    # Check if phone already exists
    existing = await db.execute(select(User).where(User.phone == body.phone))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Phone {body.phone} is already registered",
        )

    user = User(
        phone=body.phone,
        hashed_password=_pwd_ctx.hash(body.password),
        full_name=body.full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive a JWT token",
    description=(
        "Authenticate with phone + password. Returns a Bearer JWT token "
        "valid for 24 hours (configurable via JWT_EXPIRE_MINUTES env var)."
    ),
)
async def login(body: UserLogin, db: AsyncSession = Depends(get_db)):
    """
    **Login** with phone number and password.

    Returns a JWT `access_token` for use in the `Authorization: Bearer <token>` header.
    """
    result = await db.execute(select(User).where(User.phone == body.phone))
    user = result.scalar_one_or_none()

    if user is None or not _pwd_ctx.verify(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid phone number or password",
        )

    token, expires_in = create_access_token(user.id, user.phone)
    return TokenResponse(access_token=token, expires_in=expires_in)
