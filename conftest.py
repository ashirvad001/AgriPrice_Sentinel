"""
conftest.py
───────────
Shared pytest fixtures for AgriPrice Sentinel test suite.

Provides reusable stationary / non-stationary series generators and a
mock async DB session so individual test files stay DRY.
"""


import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock

# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def stationary_series() -> pd.Series:
    """White noise (n=300) — should be stationary under ADF."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0, 1, 300))


@pytest.fixture
def nonstationary_series() -> pd.Series:
    """Random walk (n=300) — should be non-stationary under ADF."""
    rng = np.random.default_rng(0)
    return pd.Series(np.cumsum(rng.normal(0, 1, 300)))


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """AsyncMock that stands in for an async SQLAlchemy session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session
