"""Optional kinematics actions shared by soft-robot models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jax import Array

from soromox.systems._dynamics import has_tangent

if TYPE_CHECKING:
    from soromox.systems.soft_robot import SoftRobot


def pose_jvp(
    primals: tuple[SoftRobot, Array, Array],
    tangents: tuple[Any, Array | None, Array | None],
) -> tuple[Array, Array] | None:
    """Select the model's arc-length action for an isolated backbone direction.

    Args:
        primals: Model, configuration, and scalar backbone coordinate.
        tangents: Model, configuration, and backbone directions. Inactive leaves
            are ``None``.

    Returns:
        Pose and directional derivative when a dedicated action applies, or
        ``None`` for joint autodiff of the active inputs.
    """
    model, q, s = primals
    dmodel, dq, ds = tangents
    if model.floating_base or has_tangent(dmodel) or dq is not None:
        return None
    if ds is None:
        pose = model._absolute_forward_kinematics(q, s)
        return pose, jnp.zeros_like(pose)
    return model._forward_kinematics_arc_length_jvp(q, s, ds)


def jacobian_jvp(
    primals: tuple[SoftRobot, Array, Array],
    tangents: tuple[Any, Array | None, Array | None],
) -> tuple[Array, Array] | None:
    """Select a geometric-Jacobian action for one active state input.

    Args:
        primals: Model, configuration, and scalar backbone coordinate.
        tangents: Model, configuration, and backbone directions. Inactive leaves
            are ``None``.

    Returns:
        Jacobian and directional derivative, or ``None`` for joint autodiff
        when model parameters, floating coordinates, or both state inputs vary.
    """
    model, q, s = primals
    dmodel, dq, ds = tangents
    if (
        model.floating_base
        or has_tangent(dmodel)
        or (dq is not None and ds is not None)
    ):
        return None
    if dq is None and ds is None:
        jacobian = model._jacobian(q, s)
        return jacobian, jnp.zeros_like(jacobian)
    if dq is None:
        jacobian, derivative = model.jacobian_and_arc_length_derivative(q, s)
        return jacobian, derivative * ds
    return model.jacobian_and_time_derivative(q, dq, s)
