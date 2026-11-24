"""PCS recurrence selection for shared dynamics derivative composition."""

import jax
import jax.numpy as jnp

from soromox.actuation.core import DirectEffort, IdentityActuator, IdentityTransmission
from soromox.actuation.threadlike import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import _dynamics
from soromox.systems._dynamics import ad_terms_jvp, has_tangent, instantiate

FUSED_STATE_MAX_VELOCITIES = 64


def supports_analytical_actuator(actuator) -> bool:
    """The closed form assumes direct effort and a frictionless fixed route."""
    return type(actuator.effort_model) is DirectEffort and (
        (
            type(actuator) is IdentityActuator
            and type(actuator.transmission) is IdentityTransmission
        )
        or (
            type(actuator) is ThreadlikeActuator
            and type(actuator.transmission.routing) is ThreadlikeRouting
            and actuator.transmission.friction.is_frictionless
        )
    )


def terms_jvp(primals, tangents):
    """Share the primal assembly for mixed state and model-parameter tangents."""
    robot, q, v = primals
    drobot, dq, dv = tangents
    if (
        has_tangent(drobot)
        or not robot._supports_analytical_inertia()
        or (dq is None and dv is not None)
    ):
        # With configuration held fixed, AD can omit configuration recurrence,
        # gravity derivatives, and inertia derivatives entirely.
        return ad_terms_jvp(primals, tangents)
    if dq is None and dv is None:
        values = robot._assemble_dynamics_terms(q, v)
        return values, jax.tree.map(jnp.zeros_like, values)
    values = robot._dynamics_terms_with_state_pushforward(
        q, v, instantiate(dq, q), instantiate(dv, v)
    )
    return values[:3], values[3:]


def _inertial_action(robot):
    def action(q, v, acceleration, dq, dv):
        return robot._inertial_force_state_pushforward(
            q, v, acceleration, dq[:, None], dv[:, None]
        )[:, 0]

    return action


def _force_action(robot):
    def action(q, v, u, dq, dv):
        return (
            -robot.K_active @ dq
            - robot.D_active @ dv
            + robot._actuation_force_state_pushforward(q, u, dq, v, dv)
            + robot._passive_force_state_pushforward(q, v, dq, dv)
        )

    return action


def _select_rule(compose, primals, tangents, *, forward):
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
                inertial_state_action=_inertial_action(robot),
                **force_options,
            ),
            default=lambda: compose(primals, tangents),
        )
    return compose(primals, tangents, terms_action=terms_jvp, **force_options)


def forward_dynamics_jvp(primals, tangents):
    return _select_rule(_dynamics.forward_dynamics_jvp, primals, tangents, forward=True)


def inverse_dynamics_jvp(primals, tangents):
    return _select_rule(
        _dynamics.inverse_dynamics_jvp, primals, tangents, forward=False
    )
