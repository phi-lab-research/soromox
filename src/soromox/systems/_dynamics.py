"""Derivative composition shared by soft-robot dynamics implementations.

Model adapters supply directional actions for their analytical recurrences.
The composition rules share primal terms and one inertia factorization across
active directions. Missing actions use JAX differentiation of the assembly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_solve, lu_factor, lu_solve

if TYPE_CHECKING:
    from soromox.systems.soft_robot import SoftRobot

Terms = tuple[Array, Array, Array]
TermsPrimals = tuple["SoftRobot", Array, Array]
TermsTangents = tuple[Any, Array | None, Array | None]
ForwardPrimals = tuple["SoftRobot", Array | float, Array, tuple | None]
ForwardTangents = tuple[Any, Array | None, Array | None, tuple | None]
InversePrimals = tuple["SoftRobot", Array, Array, Array]
InverseTangents = tuple[Any, Array | None, Array | None, Array | None]
TermsAction = Callable[[TermsPrimals, TermsTangents], tuple[Terms, Terms]]
StateTermsAction = Callable[[Array, Array, Array, Array], tuple[Array, ...]]
InertialStateAction = Callable[[Array, Array, Array, Array, Array], Array]
ForceStateAction = Callable[[Array, Array, Array, Array, Array], Array]


class InertiaFactors(NamedTuple):
    """CPU Cholesky or accelerator LU factors with the same PyTree shape."""

    matrix: jax.Array
    pivots: jax.Array


def factor_inertia(mass: jax.Array) -> InertiaFactors:
    """Factor inertia for reuse by primal and directional solves.

    CPU uses Cholesky of the symmetrized matrix; accelerators use LU.

    Args:
        mass: Positive-definite inertia, shape ``(nv, nv)``.

    Returns:
        Factor storage and pivot indices for the compilation target.
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
    """Solve with previously computed inertia factors.

    Args:
        factors: Output of :func:`factor_inertia`.
        rhs: Right-hand side, shape ``(nv,)`` or ``(nv, k)``.

    Returns:
        Solution with the same shape as ``rhs``.
    """
    return jax.lax.platform_dependent(
        factors,
        rhs,
        cpu=lambda factor, vector: cho_solve((factor.matrix, True), vector),
        default=lambda factor, vector: lu_solve(factor, vector),
    )


def has_tangent(tree: Any) -> bool:
    """Determine symbolic activity of a filtered tangent tree.

    Args:
        tree: Tangent PyTree with ``None`` for inactive leaves.

    Returns:
        Whether any leaf is active, independent of its numerical value.
    """
    return any(x is not None for x in jax.tree.leaves(tree))


def instantiate(tangent: Array | None, primal: Array) -> Array:
    """Materialize an inactive array direction as zero.

    Args:
        tangent: Array direction or ``None`` for an inactive input.
        primal: Array supplying the shape and dtype.

    Returns:
        The supplied direction, or a zero array matching ``primal``.
    """
    return jnp.zeros_like(primal) if tangent is None else tangent


def ad_terms_jvp(primals: TermsPrimals, tangents: TermsTangents) -> tuple[Terms, Terms]:
    """Evaluate one joint autodiff assembly for active state and parameter inputs.

    Args:
        primals: Model, configuration, and generalized velocity.
        tangents: Corresponding directions; inactive leaves are ``None``.

    Returns:
        Primal ``(M, Cqd, G)`` and their array-valued directional derivatives.
    """
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
    primals: ForwardPrimals,
    tangents: ForwardTangents,
    *,
    terms_action: TermsAction = ad_terms_jvp,
    inertial_state_action: InertialStateAction | None = None,
    force_state_action: ForceStateAction | None = None,
) -> tuple[Array, Array]:
    """Differentiate ``M a = applied - c - g`` with one reusable factorization.

    Args:
        primals: Model, time, packed state, and optional actuation arguments.
        tangents: Corresponding directions; inactive leaves are ``None``.
        terms_action: Bound composition returning primal and directional ``(M, Cqd, G)``.
        inertial_state_action: Optional ``(q, v, a, dq, dv)`` action for contracted inertial effort.
        force_state_action: Optional ``(q, v, u, dq, dv)`` action for net applied force.

    Returns:
        Primal packed state derivative and its direction. Parameter inputs use joint
        autodiff; symbolically inactive inputs omit their contributions.
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
    primals: InversePrimals,
    tangents: InverseTangents,
    *,
    terms_action: TermsAction = ad_terms_jvp,
    inertial_state_action: InertialStateAction | None = None,
) -> tuple[Array, Array]:
    """Compose required-effort derivatives from inertial and passive forces.

    Args:
        primals: Model, configuration, velocity, and acceleration.
        tangents: Corresponding directions; inactive leaves are ``None``.
        terms_action: Composition returning primal and directional ``(M, Cqd, G)``.
        inertial_state_action: Optional ``(q, v, a, dq, dv)`` action for contracted inertial effort.

    Returns:
        Required effort and its direction, each of shape ``(num_velocities,)``.
        Acceleration-only requests reuse the primal inertia.
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


def terms_jvp(
    primals: TermsPrimals,
    tangents: TermsTangents,
    *,
    state_action: StateTermsAction | None = None,
) -> tuple[Terms, Terms]:
    """Compose optional state actions with joint parameter differentiation.

    Args:
        primals: Model, configuration, and generalized velocity.
        tangents: Corresponding directions, with ``None`` for inactive leaves.
        state_action: Optional bound recurrence returning ``M, c, g, dM, dc, dg``
            for ``q, v, dq, dv``. A missing action selects autodiff.

    Returns:
        Primal dynamics terms and their directional derivatives. Parameter and
        velocity-only requests use one joint autodiff assembly.
    """
    robot, q, v = primals
    drobot, dq, dv = tangents
    if has_tangent(drobot) or state_action is None or (dq is None and dv is not None):
        return ad_terms_jvp(primals, tangents)
    if dq is None and dv is None:
        values = robot._assemble_dynamics_terms(q, v)
        return values, jax.tree.map(jnp.zeros_like, values)
    values = state_action(q, v, instantiate(dq, q), instantiate(dv, v))
    return values[:3], values[3:]


def inertial_state_action(
    pushforward: InertialStateAction,
    q: Array,
    v: Array,
    acceleration: Array,
    dq: Array,
    dv: Array,
) -> Array:
    """Apply a model's matrix-direction recurrence to one state direction.

    Args:
        pushforward: Bound recurrence accepting configuration/velocity direction
            matrices and returning inertial-force direction columns.
        q: Configuration, shape ``(nq,)``.
        v: Velocity, shape ``(nv,)``.
        acceleration: Fixed acceleration, shape ``(nv,)``.
        dq: Configuration direction, shape ``(nq,)``.
        dv: Velocity direction, shape ``(nv,)``.

    Returns:
        Inertial-force direction, shape ``(nv,)``.
    """
    return pushforward(q, v, acceleration, dq[:, None], dv[:, None])[:, 0]
