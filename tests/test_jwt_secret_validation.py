import importlib
import os
import sys
from unittest.mock import patch

import pytest
from pydantic import ValidationError


_BASE_ENV = {
    "DATABASE_URL": "sqlite+aiosqlite:///./test.db",
}

_APP_MODULES = [
    "app.app",
    "app.api.deps",
    "app.database",
    "app.logger",
    "app.config",
]


def _clear_app_modules() -> None:
    for module_name in _APP_MODULES:
        sys.modules.pop(module_name, None)


def _load_settings(**overrides):
    env = {**_BASE_ENV, **overrides}
    with patch.dict(os.environ, env, clear=True):
        _clear_app_modules()
        from app.config import Settings

        return Settings(_env_file=None)


def _import_app_with_env(**overrides):
    env = {**_BASE_ENV, **overrides}
    with patch.dict(os.environ, env, clear=True):
        _clear_app_modules()
        return importlib.import_module("app.app")


class TestSecretKeyValidation:
    def test_missing_secret_key_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            _load_settings()

        messages = [error.get("msg", "") for error in exc_info.value.errors()]
        assert any("SECRET_KEY environment variable is not set" in message for message in messages)

    def test_empty_secret_key_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            _load_settings(SECRET_KEY="")

        messages = [error.get("msg", "") for error in exc_info.value.errors()]
        assert any("SECRET_KEY environment variable is empty" in message for message in messages)

    def test_short_secret_key_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            _load_settings(SECRET_KEY="too-short")

        messages = [error.get("msg", "") for error in exc_info.value.errors()]
        assert any("SECRET_KEY must be at least 32 characters long" in message for message in messages)

    def test_valid_secret_key_passes(self):
        settings = _load_settings(SECRET_KEY="a" * 64)
        assert len(settings.SECRET_KEY) == 64


class TestAppStartupValidation:
    def test_app_import_fails_when_secret_key_missing(self):
        with pytest.raises(RuntimeError, match="SECRET_KEY environment variable is not set"):
            _import_app_with_env()

    def test_app_import_fails_when_secret_key_empty(self):
        with pytest.raises(RuntimeError, match="SECRET_KEY environment variable is empty"):
            _import_app_with_env(SECRET_KEY="   ")

    def test_startup_error_does_not_echo_secret_value(self):
        short_secret = "leaky_secret_value"

        with pytest.raises(RuntimeError) as exc_info:
            _import_app_with_env(SECRET_KEY=short_secret)

        assert short_secret not in str(exc_info.value)
        assert "at least 32 characters" in str(exc_info.value)
