"""
api/deps.py
-----------
FastAPI dependency injection: DB session, Redis client, JWT-based current user.
"""

import json
from typing import AsyncGenerator, Optional
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import AsyncSessionLocal, User
from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Config (validated by Pydantic on startup)
SECRET_KEY = settings.SECRET_KEY
JWT_ALGORITHM = settings.JWT_ALGORITHM
JWT_EXPIRE_MINUTES = settings.JWT_EXPIRE_MINUTES

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session, auto-closed after request."""
    async with AsyncSessionLocal() as session:
        yield session


_redis_client = None


async def init_redis():
    """Initialise a global aioredis connection pool (call at app startup)."""
    global _redis_client
    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await _redis_client.ping()
        logger.info(f"Redis connected -> {settings.REDIS_URL}")
    except Exception as e:
        _redis_client = None
        logger.warning(f"Redis unavailable ({e}) -> caching disabled")


async def close_redis():
    """Close the Redis pool at app shutdown."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None


def get_redis():
    """Return the global Redis client (may be None if unavailable)."""
    return _redis_client


async def cache_get(key: str) -> Optional[dict]:
    """Read a JSON value from Redis cache."""
    if _redis_client is None:
        return None
    try:
        raw = await _redis_client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def cache_set(key: str, value: dict, ttl: int = 3600):
    """Write a JSON value to Redis cache with TTL in seconds."""
    if _redis_client is None:
        return
    try:
        await _redis_client.set(key, json.dumps(value, default=str), ex=ttl)
    except Exception:
        pass


def create_access_token(user_id: int, phone: str) -> tuple[str, int]:
    """Create a JWT access token. Returns (token, expires_in_seconds)."""
    expires_delta = timedelta(minutes=JWT_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + expires_delta
    payload = {
        "sub": str(user_id),
        "phone": phone,
        "exp": expire,
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, int(expires_delta.total_seconds())


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate the current user from the Authorization header."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    payload = decode_access_token(credentials.credentials)
    user_id = int(payload["sub"])

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
