"""PCS recurrence selection for shared dynamics derivative composition."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import jax
from jax import Array

from soromox.systems import _dynamics
from soromox.systems._dynamics import has_tangent

if TYPE_CHECKING:
    from soromox.systems.pcs.pcs import PCS

FUSED_STATE_MAX_VELOCITIES = 64


def terms_jvp(primals: _dynamics.TermsPrimals, tangents: _dynamics.TermsTangents):
    """Supply the supported PCS recurrence to shared term composition.

    Args:
        primals: PCS model, configuration, and generalized velocity.
        tangents: Corresponding filtered model and state directions.

    Returns:
        Dynamics terms and their directional derivatives.
    """
    robot = primals[0]
    action = (
        robot._dynamics_terms_with_state_pushforward
        if robot._supports_analytical_inertia()
        else None
    )
    return _dynamics.terms_jvp(primals, tangents, state_action=action)


def _force_action(robot: PCS) -> _dynamics.ForceStateAction:
    """Bind the PCS material, actuation, and passive state actions.

    Args:
        robot: Fixed-base PCS model supporting the state recurrences.

    Returns:
        Callable accepting ``q, v, u, dq, dv`` and returning net applied-force direction.
    """

    def action(q, v, u, dq, dv):
        return (
            -robot.K_active @ dq
            - robot.D_active @ dv
            + robot._actuation_force_state_pushforward(q, u, dq, v, dv)
            + robot._passive_force_state_pushforward(q, v, dq, dv)
        )

    return action


def _select_rule(compose: Callable, primals: tuple, tangents: tuple, *, forward: bool):
    """Select the PCS recurrence for the requested inputs and compilation target.

    Args:
        compose: Shared forward- or inverse-dynamics composition.
        primals: Model and arguments of the requested dynamics operation.
        tangents: Corresponding filtered directions.
        forward: Whether the request is forward dynamics.

    Returns:
        Primal output and its direction. Large CPU requests use a contracted
        recurrence; accelerators use the shared term action.
    """
    robot = primals[0]
    parameter_active = has_tangent(tangents[0])
    state_active = (
        tangents[2] is not None
        if forward
        else (tangents[1] is not None or tangents[2] is not None)
    )
    force_options = {"force_state_action": _force_action(robot)} if forward else {}
    if parameter_active or not robot._supports_analytical_inertia():
        return compose(primals, tangents)
    if state_active and robot.num_velocities > FUSED_STATE_MAX_VELOCITIES:
        # CPU benefits from the contracted recurrence. Accelerators use AD term
        # actions to reduce sequential kernel launches. Selection follows the
        # compilation target at lowering.
        return jax.lax.platform_dependent(
            cpu=lambda: compose(
                primals,
                tangents,
                inertial_state_action=partial(
                    _dynamics.inertial_state_action,
                    robot._inertial_force_state_pushforward,
                ),
                **force_options,
            ),
            default=lambda: compose(primals, tangents),
        )
    return compose(primals, tangents, terms_action=terms_jvp, **force_options)


def forward_dynamics_jvp(
    primals: _dynamics.ForwardPrimals, tangents: _dynamics.ForwardTangents
) -> tuple[Array, Array]:
    """Supply PCS actions to forward-dynamics composition.

    Args:
        primals: Model, time, packed state, and optional actuation arguments.
        tangents: Corresponding directions; inactive leaves are ``None``.

    Returns:
        Packed state derivative and its direction.
    """
    return _select_rule(_dynamics.forward_dynamics_jvp, primals, tangents, forward=True)


def inverse_dynamics_jvp(
    primals: _dynamics.InversePrimals, tangents: _dynamics.InverseTangents
) -> tuple[Array, Array]:
    """Supply PCS actions to inverse-dynamics composition.

    Args:
        primals: Model, configuration, velocity, and acceleration.
        tangents: Corresponding directions; inactive leaves are ``None``.

    Returns:
        Required generalized effort and its direction.
    """
    return _select_rule(
        _dynamics.inverse_dynamics_jvp, primals, tangents, forward=False
    )
