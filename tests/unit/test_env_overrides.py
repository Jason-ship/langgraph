"""Tests for NOVELFACTORY_* env-var type-safe overrides."""

from __future__ import annotations

import os
from unittest.mock import patch

from novelfactory.config.settings import _coerce_env


def test_coerce_bool_true():
    for val in ("true", "1", "yes", "on", "True", "YES"):
        assert _coerce_env(val, False) is True


def test_coerce_bool_false():
    for val in ("false", "0", "no", "off", "False", "NO"):
        assert _coerce_env(val, True) is False


def test_coerce_bool_invalid():
    try:
        _coerce_env("treu", False)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "expected a boolean" in str(e)


def test_coerce_int():
    assert _coerce_env("5", 3) == 5
    assert _coerce_env("0", 10) == 0


def test_coerce_float():
    assert _coerce_env("3.14", 0.0) == 3.14
    assert _coerce_env("0.5", 1.0) == 0.5


def test_coerce_str():
    assert _coerce_env("postgres", "memory") == "postgres"


def test_env_override_applies():
    from novelfactory.config.settings import Settings

    with patch.dict(os.environ, {"NOVELFACTORY_CHECKPOINT_TYPE": "postgres"}, clear=False):
        s = Settings()
        assert s.CHECKPOINT_TYPE == "postgres"


def test_env_override_does_not_set_when_missing():
    from novelfactory.config.settings import Settings

    s = Settings()
    assert s.CHECKPOINT_TYPE  # 保留默认值
