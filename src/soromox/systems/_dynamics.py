"""Derivative composition shared by soft-robot dynamics implementations.

Model adapters supply directional actions for their analytical recurrences.
The composition rules share primal terms and one inertia factorization across
active directions. Missing actions use JAX differentiation of the assembly.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, lu_factor, lu_solve


class InertiaFactors(NamedTuple):
    """CPU Cholesky or accelerator LU factors with the same PyTree shape."""

    matrix: jax.Array
    pivots: jax.Array


def factor_inertia(mass: jax.Array) -> InertiaFactors:
    """Factor inertia once for reuse by primal and directional solves.

    Match ``SoftRobot._solve_inertia``: CPU uses Cholesky of the symmetrized
    inertia; accelerator backends use LU. Dispatch follows the compilation
    target through ``lax.platform_dependent``.
    """

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


def has_tangent(tree) -> bool:
    """Check symbolic activity, never the numerical values of a tangent."""
    return any(x is not None for x in jax.tree.leaves(tree))


def instantiate(tangent, primal):
    return jnp.zeros_like(primal) if tangent is None else tangent


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


def forward_dynamics_jvp(
    primals,
    tangents,
    *,
    terms_action=ad_terms_jvp,
    inertial_state_action=None,
    force_state_action=None,
):
    """Differentiate ``M a = applied - c - g`` with reusable factors.

    ``terms_action`` returns values and directional derivatives of ``(M,c,g)``.
    A contracted ``inertial_state_action(q,v,a,dq,dv)`` may replace those
    derivatives for state-only requests. ``force_state_action(q,v,u,dq,dv)``
    supplies the applied-force state derivative. Both optional actions are
    bound to the primal model; parameter directions use one joint AD assembly.
    Activity is determined from symbolic ``None`` tangents at tracing time.
    """
    robot, _, y, args = primals
    drobot, _, dy, dargs = tangents
    q, v, auxiliary = robot.split_state(y)
    dq, dv, _ = (None, None, None) if dy is None else robot.split_state(dy)
    u, external = robot._actuation_inputs(args, y.dtype)
    du = None if dargs is None else dargs[0]
    dext = dargs[1] if dargs is not None and len(dargs) == 2 else None
    parameter_active = has_tangent(drobot)
    if parameter_active:
        terms_action = ad_terms_jvp
        inertial_state_action = force_state_action = None
    contracted = dy is not None and inertial_state_action is not None
    if (dy is None and not parameter_active) or contracted:
        mass, convective, gravity = robot._assemble_dynamics_terms(q, v)
    else:
        (mass, convective, gravity), (dm, dc, dg) = terms_action(
            (robot, q, v), (drobot, dq, dv)
        )

    if dy is not None and force_state_action is not None:
        force = robot._applied_force(q, v, u, external)
        dforce = force_state_action(q, v, u, dq, dv)
        if du is not None:
            _, control_direction = jax.jvp(
                lambda control: robot._actuation_force(q, control, v), (u,), (du,)
            )
            dforce = dforce + control_direction
        if dext is not None:
            dforce = dforce + dext
    else:
        force, dforce = eqx.filter_jvp(
            lambda model, q, v, u, external: model._applied_force(q, v, u, external),
            (robot, q, v, u, external),
            (drobot, dq, dv, du, dext),
        )
        dforce = instantiate(dforce, force)

    factors = factor_inertia(mass)
    acceleration = solve_factored(factors, force - convective - gravity)
    if contracted:
        inertial_direction = inertial_state_action(q, v, acceleration, dq, dv)
    elif dy is None and not parameter_active:
        inertial_direction = jnp.zeros_like(v)
    else:
        inertial_direction = dm @ acceleration + dc + dg
    da = solve_factored(factors, dforce - inertial_direction)
    qdot, dqdot = eqx.filter_jvp(
        lambda model, q, v: model.configuration_derivative(q, v),
        (robot, q, v),
        (drobot, dq, dv),
    )
    return (
        robot.pack_state(qdot, acceleration, jnp.zeros_like(auxiliary)),
        robot.pack_state(instantiate(dqdot, qdot), da, jnp.zeros_like(auxiliary)),
    )


def inverse_dynamics_jvp(
    primals,
    tangents,
    *,
    terms_action=ad_terms_jvp,
    inertial_state_action=None,
):
    """Compose required-effort derivatives from model terms and passive forces.

    The optional actions have the same contracts as in ``forward_dynamics_jvp``.
    Acceleration-only requests reuse the primal inertia directly.
    """
    robot, q, v, acceleration = primals
    drobot, dq, dv, da = tangents
    parameter_active = has_tangent(drobot)
    state_active = dq is not None or dv is not None
    if parameter_active:
        terms_action = ad_terms_jvp
        inertial_state_action = None
    if not parameter_active and (not state_active or inertial_state_action is not None):
        mass, convective, gravity = robot._assemble_dynamics_terms(q, v)
        direction = (
            inertial_state_action(
                q, v, acceleration, instantiate(dq, q), instantiate(dv, v)
            )
            if state_active
            else jnp.zeros_like(v)
        )
    else:
        (mass, convective, gravity), (dm, dc, dg) = terms_action(
            (robot, q, v), (drobot, dq, dv)
        )
        direction = dm @ acceleration + dc + dg
    passive, dpassive = eqx.filter_jvp(
        lambda model, q, v: model.passive_force(q, v),
        (robot, q, v),
        (drobot, dq, dv),
    )
    direction = direction + instantiate(dpassive, passive)
    if da is not None:
        direction = direction + mass @ da
    return mass @ acceleration + convective + gravity + passive, direction
