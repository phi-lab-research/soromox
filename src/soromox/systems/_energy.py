"""Energy derivative composition using the model's force and inertia hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
from jax import Array

from soromox.systems._dynamics import has_tangent

if TYPE_CHECKING:
    from soromox.systems.soft_robot import SoftRobot

EnergyKind = Literal["gravitational", "elastic", "kinetic", "potential", "total"]
EnergyPrimals = tuple["SoftRobot", Array, Array | None, EnergyKind]
EnergyTangents = tuple[Any, Array | None, Array | None, None]


def evaluate(
    model: SoftRobot, q: Array, velocity: Array | None, kind: EnergyKind
) -> Array:
    """Evaluate an energy using the model's configuration layout.

    Args:
        model: Soft-robot model supplying protected energy equations.
        q: Configuration, shape ``(num_coordinates,)``.
        velocity: Velocity for kinetic/total energy; otherwise ``None``.
        kind: One of gravitational, elastic, kinetic, potential, or total.

    Returns:
        Scalar energy, including the appropriate floating-base contributions.
    """
    if model.floating_base:
        if kind == "elastic":
            return model._elastic_energy(model.split_configuration(q)[1])
        if kind in ("potential", "total"):
            value = model._gravitational_energy(q) + model._elastic_energy(
                model.split_configuration(q)[1]
            )
            return (
                value + model._kinetic_energy(q, velocity) if kind == "total" else value
            )
    primal = getattr(model, f"_{kind}_energy")
    return primal(q, velocity) if kind in ("kinetic", "total") else primal(q)


def energy_jvp(
    primals: EnergyPrimals, tangents: EnergyTangents
) -> tuple[Array, Array] | None:
    """Compose active energy contributions using force and inertia identities.

    Args:
        primals: Model, configuration, optional velocity, and energy kind.
        tangents: Corresponding directions; inactive leaves are ``None``.

    Returns:
        Scalar energy and its direction, or ``None`` for joint autodiff when
        model parameters or floating-base coordinates are involved.
    """
    model, q, velocity, kind = primals
    dmodel, dq, dv, _ = tangents
    if model.floating_base or has_tangent(dmodel):
        return None
    value = evaluate(*primals)
    direction = jnp.zeros_like(value)
    if kind in ("kinetic", "total"):
        if dq is not None:
            direction = direction + model._kinetic_energy_q_jvp_tangent(q, velocity, dq)
        if dv is not None:
            direction = direction + model._kinetic_energy_qd_jvp_tangent(
                q, velocity, dv
            )
    if dq is not None:
        if kind in ("gravitational", "potential", "total"):
            direction = direction + model._gravitational_energy_q_jvp_tangent(q, dq)
        if kind in ("elastic", "potential", "total"):
            direction = direction + model.elastic_force(q) @ dq
    return value, direction
