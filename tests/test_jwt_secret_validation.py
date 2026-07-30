"""
tests/test_jwt_secret_validation.py
────────────────────────────────────
Unit tests asserting that the app refuses to start when JWT_SECRET is
missing, empty, or too short.

These tests instantiate Settings directly (bypassing the lru_cache singleton)
with a fully controlled environment to verify the Pydantic validation.
"""

import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError


# ── Helpers ──────────────────────────────────────────────────────────────────

# Minimal env vars that satisfy every *other* required field so we only
# test JWT_SECRET validation in isolation.
_BASE_ENV = {
    "DATABASE_URL": "sqlite+aiosqlite:///./test.db",
}

def _make_settings(**overrides):
    """Create a fresh Settings instance with controlled env vars."""
    env = {**_BASE_ENV, **overrides}
    # Prevent reading the real .env file by pointing to a non-existent file
    with patch.dict(os.environ, env, clear=True):
        from app.config import Settings
        return Settings(_env_file=None)  # skip .env file


# ── Tests ────────────────────────────────────────────────────────────────────

class TestJWTSecretMissing:
    """App must crash at startup when JWT_SECRET is absent."""

    def test_missing_jwt_secret_raises(self):
        """Settings() must raise ValidationError when JWT_SECRET is not set."""
        with pytest.raises(ValidationError) as exc_info:
            _make_settings()  # no JWT_SECRET provided
        errors = exc_info.value.errors()
        jwt_errors = [e for e in errors if "JWT_SECRET" in str(e.get("loc", ""))]
        assert len(jwt_errors) >= 1, "Expected a validation error for JWT_SECRET"

    def test_empty_jwt_secret_raises(self):
        """An empty-string JWT_SECRET is treated as missing."""
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(JWT_SECRET="")
        errors = exc_info.value.errors()
        jwt_errors = [e for e in errors if "JWT_SECRET" in str(e.get("loc", ""))]
        assert len(jwt_errors) >= 1

    def test_whitespace_only_jwt_secret_raises(self):
        """A whitespace-only JWT_SECRET is treated as missing."""
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(JWT_SECRET="   ")
        errors = exc_info.value.errors()
        jwt_errors = [e for e in errors if "JWT_SECRET" in str(e.get("loc", ""))]
        assert len(jwt_errors) >= 1


class TestJWTSecretTooShort:
    """App must crash when JWT_SECRET is below the 32-char minimum."""

    def test_short_jwt_secret_raises(self):
        """A 10-character secret must be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(JWT_SECRET="tooshort!!")  # 10 chars
        errors = exc_info.value.errors()
        jwt_errors = [e for e in errors if "JWT_SECRET" in str(e.get("loc", ""))]
        assert len(jwt_errors) >= 1

    def test_exactly_31_chars_raises(self):
        """31 characters is just below the threshold — must fail."""
        with pytest.raises(ValidationError):
            _make_settings(JWT_SECRET="a" * 31)


class TestJWTSecretValid:
    """A properly-configured JWT_SECRET should pass validation."""

    def test_valid_64_char_hex_secret(self):
        """A 64-char hex string (output of `openssl rand -hex 32`) should work."""
        settings = _make_settings(
            JWT_SECRET="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        )
        assert len(settings.JWT_SECRET) == 64

    def test_exactly_32_chars_passes(self):
        """The minimum length (32 chars) should be accepted."""
        settings = _make_settings(JWT_SECRET="a" * 32)
        assert len(settings.JWT_SECRET) == 32


class TestSecretNotLogged:
    """The validator must never include the actual secret value in errors."""

    def test_error_message_does_not_contain_secret(self):
        """Ensure our custom validator message doesn't leak the secret value.

        Note: Pydantic's ValidationError repr always includes input_value for
        debugging.  We verify that *our* error message (the 'msg' field) never
        contains the actual secret.
        """
        short_secret = "leaky_secret_value"
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(JWT_SECRET=short_secret)
        errors = exc_info.value.errors()
        for error in errors:
            # 'msg' contains our custom ValueError text
            assert short_secret not in error.get("msg", ""), (
                "The actual secret value must NEVER appear in validator error messages"
            )
