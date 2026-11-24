import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


@pytest.fixture
def analytical_derivatives():
    """Opt in explicitly for tests of analytical paths, restoring trace state."""
    import jax

    from soromox import custom_jvp_mode

    jax.clear_caches()
    try:
        with custom_jvp_mode(True):
            yield
    finally:
        jax.clear_caches()
