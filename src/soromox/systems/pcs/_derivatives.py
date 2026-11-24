"""Private derivative composition for PCS dynamics.

The implicit acceleration solve shares a factorization across the primal and
all active tangent contributions. State recurrences remain in PCS; this module
owns composition, activity selection, and the parameter fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, lu_factor, lu_solve

from soromox.actuation.core import DirectEffort, IdentityActuator, IdentityTransmission
from soromox.actuation.threadlike import ThreadlikeActuator, ThreadlikeRouting

if TYPE_CHECKING:
    from soromox.systems.pcs.pcs import PCS


FUSED_STATE_MAX_VELOCITIES = 64


class InertiaFactors(NamedTuple):
    """CPU Cholesky or accelerator LU factors with the same PyTree shape."""

    matrix: jax.Array
    pivots: jax.Array


def factor_inertia(mass: jax.Array) -> InertiaFactors:
    """Preserve the existing CPU symmetrization and accelerator solve policy."""

    def cholesky(matrix):
        return InertiaFactors(
            jnp.linalg.cholesky(0.5 * (matrix + matrix.T)),
            jnp.arange(matrix.shape[0], dtype=jnp.int32),
        )

    def generic(matrix):
        return InertiaFactors(*lu_factor(matrix))

    return jax.lax.platform_dependent(mass, cpu=cholesky, default=generic)


def solve_factored(factors: InertiaFactors, rhs: jax.Array) -> jax.Array:
    """Apply saved factors to a vector or several right-hand sides."""
    return jax.lax.platform_dependent(
        factors,
        rhs,
        cpu=lambda factor, vector: cho_solve((factor.matrix, True), vector),
        default=lambda factor, vector: lu_solve(factor, vector),
    )


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


def has_tangent(tree) -> bool:
    """Check symbolic activity, never the numerical values of a tangent."""
    return any(x is not None for x in jax.tree.leaves(tree))


def instantiate(tangent, primal):
    return jnp.zeros_like(primal) if tangent is None else tangent


def controls(robot: PCS, args, dtype):
    if args is not None and len(args) not in (1, 2):
        raise ValueError("actuation_args must be a tuple of length 1 or 2.")
    u = None if args is None else args[0]
    external = args[1] if args is not None and len(args) == 2 else None
    return (
        jnp.zeros((robot.num_actuators,), dtype=dtype) if u is None else u,
        jnp.zeros((robot.num_velocities,), dtype=dtype)
        if external is None
        else external,
    )


def applied_force(robot: PCS, q, v, u, external):
    return (
        robot._actuation_force(q, u, v)
        + external
        - robot.elastic_force(q)
        - robot.damping_matrix(q) @ v
    )


def ad_terms_jvp(primals, tangents):
    """One AD assembly for the union of active state and parameter leaves."""
    values, directions = eqx.filter_jvp(
        lambda model, position, velocity: model._assemble_dynamics_terms(
            position, velocity
        ),
        primals,
        tangents,
    )
    return values, jax.tree.map(
        instantiate, directions, values, is_leaf=lambda x: x is None
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


def forward_dynamics_jvp(primals, tangents):
    robot, _, _, _ = primals
    drobot, _, dy, _ = tangents
    if (
        robot._supports_analytical_inertia()
        and dy is not None
        and not has_tangent(drobot)
        and robot.num_velocities > FUSED_STATE_MAX_VELOCITIES
    ):
        # The contracted recurrence is efficient on CPU but sequential launches
        # are costly on accelerators. Selection happens at lowering, not by
        # inspecting the current default device or numerical tangent values.
        return jax.lax.platform_dependent(
            cpu=lambda: _forward_dynamics_jvp(primals, tangents),
            default=lambda: _forward_dynamics_jvp(
                primals, tangents, general_terms=True
            ),
        )
    return _forward_dynamics_jvp(primals, tangents)


def _forward_dynamics_jvp(primals, tangents, *, general_terms=False):
    robot, t, y, args = primals
    drobot, _, dy, dargs = tangents
    if not robot._supports_analytical_inertia():
        return eqx.filter_jvp(
            lambda model, time, state, inputs: model._forward_dynamics(
                time, state, inputs
            ),
            primals,
            tangents,
        )
    q, v = jnp.split(y, 2)
    dq, dv = (None, None) if dy is None else jnp.split(dy, 2)
    u, external = controls(robot, args, y.dtype)
    du = None if dargs is None else dargs[0]
    dext = dargs[1] if dargs is not None and len(dargs) == 2 else None
    joint_terms = has_tangent(drobot) or general_terms

    if joint_terms:
        # One evaluation for the union of state and parameter directions avoids
        # assembling the same primal separately for each differentiated group.
        (mass, convective, gravity), (dmass, dc, dg) = ad_terms_jvp(
            (robot, q, v), (drobot, dq, dv)
        )
        force, dforce = eqx.filter_jvp(
            applied_force, (robot, q, v, u, external), (drobot, dq, dv, du, dext)
        )
        dforce = instantiate(dforce, force)
    else:
        if dy is None or robot.num_velocities > FUSED_STATE_MAX_VELOCITIES:
            mass, convective, gravity = robot._assemble_dynamics_terms(q, v)
        else:
            (mass, convective, gravity), (dmass, dc, dg) = terms_jvp(
                (robot, q, v), (drobot, dq, dv)
            )
        if dy is None:
            force, dforce = eqx.filter_jvp(
                applied_force,
                (robot, q, v, u, external),
                (drobot, None, None, du, dext),
            )
            dforce = instantiate(dforce, force)
        else:
            force = applied_force(robot, q, v, u, external)
            dforce = jnp.zeros_like(v)
        if dy is not None:
            dforce = (
                -robot.K_active @ dq
                - robot.D_active @ dv
                + robot._actuation_force_state_pushforward(q, u, dq, v, dv)
                + robot._passive_force_state_pushforward(q, v, dq, dv)
            )
        if dy is not None and du is not None:
            _, control_direction = jax.jvp(
                lambda control: robot._actuation_force(q, control, v), (u,), (du,)
            )
            dforce = dforce + control_direction
        if dy is not None and dext is not None:
            dforce = dforce + dext

    factors = factor_inertia(mass)
    acceleration = solve_factored(factors, force - convective - gravity)
    if not joint_terms and dy is None:
        inertial_direction = jnp.zeros_like(v)
    elif not joint_terms and robot.num_velocities > FUSED_STATE_MAX_VELOCITIES:
        inertial_direction = robot.inverse_dynamics_state_pushforward(
            q, v, acceleration, dq[:, None], dv[:, None]
        )[:, 0]
    else:
        inertial_direction = dmass @ acceleration + dc + dg
    da = solve_factored(factors, dforce - inertial_direction)
    return jnp.concatenate((v, acceleration)), jnp.concatenate((instantiate(dv, v), da))


def forward_dynamics_reference(robot: PCS, t, y, args):
    """Independent JAX equations; bypass analytical dynamics-term boundaries."""
    del t
    q, v, auxiliary = robot.split_state(y)
    u, external = controls(robot, args, y.dtype)
    mass, convective, gravity = robot._assemble_dynamics_terms(q, v)
    acceleration = robot._solve_inertia(
        mass, applied_force(robot, q, v, u, external) - convective - gravity
    )
    return robot.pack_state(
        robot.configuration_derivative(q, v), acceleration, jnp.zeros_like(auxiliary)
    )


def passive_force(robot: PCS, q, v):
    return robot.elastic_force(q) + robot.damping_matrix(q) @ v


def inverse_dynamics_reference(robot: PCS, q, v, acceleration):
    mass, convective, gravity = robot._assemble_dynamics_terms(q, v)
    return mass @ acceleration + convective + gravity + passive_force(robot, q, v)


@eqx.filter_custom_jvp
def inverse_dynamics(robot: PCS, q, v, acceleration):
    return inverse_dynamics_reference(robot, q, v, acceleration)


@inverse_dynamics.def_jvp
def _inverse_dynamics_rule(primals, tangents):
    robot, _, _, _ = primals
    drobot, dq, dv, _ = tangents
    if (
        robot._supports_analytical_inertia()
        and (dq is not None or dv is not None)
        and not has_tangent(drobot)
        and robot.num_velocities > FUSED_STATE_MAX_VELOCITIES
    ):
        return jax.lax.platform_dependent(
            cpu=lambda: _inverse_dynamics_jvp(primals, tangents),
            default=lambda: _inverse_dynamics_jvp(
                primals, tangents, general_terms=True
            ),
        )
    return _inverse_dynamics_jvp(primals, tangents)


def _inverse_dynamics_jvp(primals, tangents, *, general_terms=False):
    robot, q, v, acceleration = primals
    drobot, dq, dv, da = tangents
    parameter_active = has_tangent(drobot)
    state_active = dq is not None or dv is not None
    large_state = (
        not parameter_active
        and not general_terms
        and state_active
        and robot._supports_analytical_inertia()
        and robot.num_velocities > FUSED_STATE_MAX_VELOCITIES
    )
    if large_state:
        mass, convective, gravity = robot._assemble_dynamics_terms(q, v)
        direction = robot.inverse_dynamics_state_pushforward(
            q, v, acceleration, instantiate(dq, q)[:, None], instantiate(dv, v)[:, None]
        )[:, 0]
    else:
        action = ad_terms_jvp if general_terms else terms_jvp
        (mass, convective, gravity), (dm, dc, dg) = action(
            (robot, q, v), (drobot, dq, dv)
        )
        direction = dm @ acceleration + dc + dg
    passive, dpassive = eqx.filter_jvp(passive_force, (robot, q, v), (drobot, dq, dv))
    direction = direction + instantiate(dpassive, passive)
    if da is not None:
        direction = direction + mass @ da
    return mass @ acceleration + convective + gravity + passive, direction
