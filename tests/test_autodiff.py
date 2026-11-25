"""Package defaults and explicit derivative-mode configuration."""

import subprocess
import sys

import pytest

from soromox import custom_jvp_enabled, custom_jvp_mode, set_custom_jvp_enabled


def test_fresh_import_defaults_to_autodiff():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from soromox import custom_jvp_enabled; assert custom_jvp_enabled() is False",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_nested_opt_in_restores_default_after_exception():
    assert custom_jvp_enabled() is False
    with pytest.raises(RuntimeError, match="test failure"), custom_jvp_mode(True):
        assert custom_jvp_enabled() is True
        with custom_jvp_mode(False):
            assert custom_jvp_enabled() is False
        assert custom_jvp_enabled() is True
        raise RuntimeError("test failure")
    assert custom_jvp_enabled() is False


def test_global_opt_in_and_reset():
    assert custom_jvp_enabled() is False
    try:
        set_custom_jvp_enabled(True)
        assert custom_jvp_enabled() is True
        set_custom_jvp_enabled(False)
        assert custom_jvp_enabled() is False
    finally:
        set_custom_jvp_enabled(False)
