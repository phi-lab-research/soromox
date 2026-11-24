"""Global autodiff configuration helpers.

Both flags are read by public APIs at trace time. Jitted functions capture the
Python value when they are traced, so change a flag before tracing or retrace
after changing it.

- ``custom_jvp_enabled`` is false by default: ordinary JAX autodiff is used.
  Enable it before tracing to opt into optional analytical derivative dispatch when
  applying JAX transformations to system methods, including shared execution
  boundaries. Explicit analytical helpers retain their requested algorithms.
  Numerical safety and implicit-solver derivative definitions remain active.
- ``strict_singularities_enabled`` selects whether the ``safe_*`` helpers in
  ``soromox.utils`` fall back at a singular input or let it propagate. Enable it
  to find out *where* a model reaches a singular configuration; leave it off for
  normal runs.

Strict mode reports singular inputs reaching a ``safe_*`` helper. It says
nothing about closed forms that avoid their singularity by branching, since
``lax.cond`` never evaluates the untaken branch. A clean strict run therefore
means "no helper was asked to substitute a value", not "no singular
configuration was visited".
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

__all__ = [
    "custom_jvp_enabled",
    "custom_jvp_mode",
    "set_custom_jvp_enabled",
    "strict_singularities_enabled",
    "strict_singularities_mode",
    "set_strict_singularities",
]

_CUSTOM_JVP_ENABLED = False
_STRICT_SINGULARITIES = False


def custom_jvp_enabled() -> bool:
    """Read the optional analytical derivative mode.

    Returns:
        Whether analytical rules are enabled. The package default is ``False``.
    """
    return _CUSTOM_JVP_ENABLED


def set_custom_jvp_enabled(enabled: bool) -> None:
    """Select optional analytical rules before tracing application functions.

    Args:
        enabled: ``True`` enables optional rules; ``False`` selects autodiff.

    Returns:
        None. Updates the process-wide setting for future traces.
    """
    global _CUSTOM_JVP_ENABLED
    _CUSTOM_JVP_ENABLED = bool(enabled)


@contextmanager
def custom_jvp_mode(enabled: bool) -> Iterator[None]:
    """Temporarily select optional analytical rules in system methods.

    Args:
        enabled: Whether to enable optional rules for traces inside the context.

    Yields:
        None. Restores the previous setting on exit, including exceptions.
    """
    previous = custom_jvp_enabled()
    set_custom_jvp_enabled(enabled)
    try:
        yield
    finally:
        set_custom_jvp_enabled(previous)


def strict_singularities_enabled() -> bool:
    """Return whether the ``safe_*`` helpers propagate singular inputs."""
    return _STRICT_SINGULARITIES


def set_strict_singularities(enabled: bool) -> None:
    """Set whether the ``safe_*`` helpers propagate singular inputs.

    When enabled, the helpers drop their fallbacks and behave like the plain
    operations they replace, so a singular input surfaces as ``NaN`` or ``inf``
    instead of a finite substitute. Use it to locate the configuration a model
    actually reaches; the fallbacks are what keep production runs finite.

    Args:
        enabled: ``True`` to propagate singular inputs, ``False`` to fall back.
    """
    global _STRICT_SINGULARITIES
    _STRICT_SINGULARITIES = bool(enabled)


@contextmanager
def strict_singularities_mode(enabled: bool = True) -> Iterator[None]:
    """Temporarily propagate singular inputs through the ``safe_*`` helpers.

    Args:
        enabled: ``True`` to propagate singular inputs, ``False`` to fall back.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from soromox.autodiff import strict_singularities_mode
        from soromox.utils import safe_norm

        jax.grad(safe_norm)(jnp.zeros(3))  # [0. 0. 0.]
        with strict_singularities_mode():
            jax.grad(safe_norm)(jnp.zeros(3))  # [nan nan nan]
        ```
    """
    previous = strict_singularities_enabled()
    set_strict_singularities(enabled)
    try:
        yield
    finally:
        set_strict_singularities(previous)
