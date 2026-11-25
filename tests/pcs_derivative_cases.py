"""PCS fixtures and composed derivative queries for numerical regression tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import PCS, IsotropicMaterialParams, LinkSpec, PCSStructure

Array = jax.Array
Workload = Callable[[Array, Array, Array, Array], Any]
LINK_LENGTH = 0.1
LINK_DENSITY = 1070.0
YOUNG_MODULUS = 2.0e3
SHEAR_MODULUS = 1.0e3
MATERIAL_DAMPING_COEFFICIENT = 1.0e-3


class DerivativeInputs(NamedTuple):
    """Dynamic inputs and directions shared by the regression queries."""

    y: Array
    y_direction: Array
    u: Array
    u_direction: Array
    tau_ext: Array
    system_parameters: Array
    system_parameter_direction: Array


def _strain_selector(setup: str, num_segments: int) -> Array | None:
    """Select the strain coordinates used by a regression model.

    Args:
        setup: Named physical configuration.
        num_segments: Number of constant-strain sections.

    Returns:
        Boolean strain mask, or None for all strains.
    """
    if setup == "full":
        return None
    if setup in {"bending_extension", "threadlike", "custom_inertia"}:
        per_segment = jnp.array([False, False, True, True, False, False])
    elif setup == "axial":
        per_segment = jnp.array([False, False, False, True, False, False])
    else:
        raise ValueError(f"Unknown setup: {setup}")
    return jnp.tile(per_segment, num_segments)


def _threadlike_actuator(num_segments: int) -> ThreadlikeActuator:
    """Construct the tendon actuation used by component derivative tests.

    Args:
        num_segments: Number of sections traversed by the two tendons.

    Returns:
        Frictionless actuator with fixed offset routing.
    """
    offsets = jnp.array([-0.01, 0.01])
    routing = ThreadlikeRouting.linear(
        intercept=jnp.stack(
            [jnp.zeros_like(offsets), offsets, jnp.zeros_like(offsets)], axis=1
        ),
        slope=jnp.zeros((2, 3)),
        start_segment_index=(0, 0),
        end_segment_index=(num_segments - 1, num_segments - 1),
    )
    return ThreadlikeActuator.tendons(routing)


def build_model(setup: str, num_segments: int, gauss_points: int) -> PCS:
    """Construct a PCS model for numerical regression queries.

    Args:
        setup: Full, reduced, axial, threadlike, or custom-inertia configuration.
        num_segments: Number of constant-strain sections.
        gauss_points: Quadrature nodes per section.

    Returns:
        Fixed-base PCS model using differentiable execution.
    """
    actuators = _threadlike_actuator(num_segments) if setup == "threadlike" else None
    model = PCS.from_links(
        [
            LinkSpec.circular(
                length=LINK_LENGTH,
                radius=0.02,
                density=LINK_DENSITY,
                young_modulus=YOUNG_MODULUS,
                shear_modulus=SHEAR_MODULUS,
                material_damping_coefficient=MATERIAL_DAMPING_COEFFICIENT,
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            for _ in range(num_segments)
        ],
        gravity=jnp.array([0.0, 0.0, -9.81]),
        structure=PCSStructure(
            num_gauss_points=gauss_points,
            strain_selector=_strain_selector(setup, num_segments),
        ),
        actuators=actuators,
        backend="jax",
    )
    if setup == "custom_inertia":
        model = model.update_link_params(
            tip_inertia=jnp.broadcast_to(0.01 * jnp.eye(6), (num_segments, 6, 6))
        )
    return model


def build_inputs(model: PCS) -> DerivativeInputs:
    """Construct dynamic inputs and nonzero perturbations for a model.

    Args:
        model: Fixed-base PCS regression model.

    Returns:
        State, controls, external effort, material parameters, and directions.
    """
    q = jnp.linspace(-0.025, 0.035, model.num_velocities)
    qd = jnp.linspace(0.04, -0.03, model.num_velocities)
    y = jnp.concatenate([q, qd])
    y_direction = jnp.linspace(-0.2, 0.25, 2 * model.num_velocities)
    u = jnp.linspace(0.05, 0.1, model.num_actuators)
    u_direction = jnp.linspace(-0.15, 0.2, model.num_actuators)
    tau_ext = jnp.linspace(-0.01, 0.015, model.num_velocities)
    system_parameters = jnp.concatenate(
        [
            model.L,
            model.rho,
            jnp.full((model.num_segments,), YOUNG_MODULUS, dtype=y.dtype),
        ]
    )
    system_parameter_direction = 0.01 * system_parameters
    return DerivativeInputs(
        y=y,
        y_direction=y_direction,
        u=u,
        u_direction=u_direction,
        tau_ext=tau_ext,
        system_parameters=system_parameters,
        system_parameter_direction=system_parameter_direction,
    )


def model_with_system_parameters(model: PCS, parameters: Array) -> PCS:
    """Update per-segment length, density, and Young's modulus.

    Args:
        model: Model whose structural choices are retained.
        parameters: Concatenated length, density, and Young's modulus arrays.

    Returns:
        Updated model with refreshed material-dependent quantities.
    """
    length, density, young_modulus = jnp.split(parameters, 3)
    updated = model.update_link_params(length=length, density=density)
    material = IsotropicMaterialParams(
        young_modulus=young_modulus,
        shear_modulus=jnp.full_like(young_modulus, SHEAR_MODULUS),
        material_damping_coefficient=jnp.full_like(
            young_modulus,
            MATERIAL_DAMPING_COEFFICIENT,
        ),
    )
    return updated.with_isotropic_material(material)


def rollout_final_position(
    model: PCS,
    y: Array,
    u: Array,
    tau_ext: Array,
    *,
    rollout_steps: int,
    rollout_dt: float,
    use_custom_jvp: bool,
    checkpoint: bool = False,
) -> Array:
    """Integrate a short symplectic-Euler trajectory.

    Args:
        model: Fixed-base PCS model.
        y: Initial packed state.
        u: Constant actuator inputs.
        tau_ext: Constant external generalized effort.
        rollout_steps: Number of integration steps.
        rollout_dt: Step duration.
        use_custom_jvp: Whether to call the public derivative boundary.
        checkpoint: Whether to rematerialize step intermediates in reverse mode.

    Returns:
        Final generalized position.
    """
    dt = jnp.asarray(rollout_dt, dtype=y.dtype)

    def step(y_value: Array, step_index: Array) -> tuple[Array, None]:
        t = dt * step_index
        if use_custom_jvp:
            yd = model.forward_dynamics(t, y_value, (u, tau_ext))
        else:
            yd = model._forward_dynamics(t, y_value, (u, tau_ext))
        q, qd = jnp.split(y_value, 2)
        _, qdd = jnp.split(yd, 2)
        qd_next = qd + dt * qdd
        q_next = q + dt * qd_next
        return jnp.concatenate([q_next, qd_next]), None

    final_y, _ = jax.lax.scan(
        jax.checkpoint(step) if checkpoint else step,
        y,
        jnp.arange(rollout_steps, dtype=jnp.int32),
    )
    final_q, _ = jnp.split(final_y, 2)
    return final_q


def build_workload(
    model: PCS,
    workload: str,
    y_direction: Array,
    u_direction: Array,
    system_parameter_direction: Array,
    *,
    rollout_steps: int,
    rollout_dt: float,
    use_custom_jvp: bool,
) -> Workload:
    """Construct a composed derivative regression query.

    Args:
        model: Fixed-base PCS model.
        workload: Query name selected by a regression test.
        y_direction: Packed state direction.
        u_direction: Actuation direction.
        system_parameter_direction: Length, density, and Young's modulus direction.
        rollout_steps: Number of steps for trajectory queries.
        rollout_dt: Integration step duration.
        use_custom_jvp: Whether to use public methods. The reference uses protected primals.

    Returns:
        Callable accepting state, controls, external effort, and material parameters.
    """
    t = jnp.array(0.0)

    def dynamics(
        y: Array,
        u: Array,
        tau_ext: Array,
        system_parameters: Array,
    ) -> Array:
        del system_parameters
        if use_custom_jvp:
            return model.forward_dynamics(t, y, (u, tau_ext))
        return model._forward_dynamics(t, y, (u, tau_ext))

    def rollout(
        rollout_model: PCS,
        y: Array,
        u: Array,
        tau_ext: Array,
    ) -> Array:
        return rollout_final_position(
            rollout_model,
            y,
            u,
            tau_ext,
            rollout_steps=rollout_steps,
            rollout_dt=rollout_dt,
            use_custom_jvp=use_custom_jvp,
        )

    def parameterized_dynamics(y, u, force, parameters):
        updated = model_with_system_parameters(model, parameters)
        if use_custom_jvp:
            return updated.forward_dynamics(t, y, (u, force))
        return updated._forward_dynamics(t, y, (u, force))

    if workload in {"state_vjp", "input_vjp"}:

        def product(y, u, force, parameters):
            variable = y if workload == "state_vjp" else u

            def evaluate(value):
                return (
                    dynamics(value, u, force, parameters)
                    if workload == "state_vjp"
                    else dynamics(y, value, force, parameters)
                )

            primal, pullback = jax.vjp(evaluate, variable)
            return primal, pullback(y_direction)[0]

        return product
    if workload == "mixed_jvp":
        return lambda y, u, force, parameters: jax.jvp(
            parameterized_dynamics,
            (y, u, force, parameters),
            (
                y_direction,
                u_direction,
                jnp.ones_like(force),
                system_parameter_direction,
            ),
        )
    if workload == "mixed_vjp":

        def mixed_product(y, u, force, parameters):
            primal, pullback = jax.vjp(parameterized_dynamics, y, u, force, parameters)
            return primal, pullback(y_direction)

        return mixed_product
    if workload == "state_jacrev":
        return lambda y, u, force, parameters: jax.jacrev(dynamics)(
            y, u, force, parameters
        )
    if workload == "state_directions":

        def directions(y, u, force, parameters):
            primal, action = jax.linearize(
                lambda state: dynamics(state, u, force, parameters), y
            )
            seeds = jnp.stack([jnp.roll(y_direction, i) for i in range(4)])
            return primal, jax.vmap(action)(seeds)

        return directions
    if workload == "batched_state_jvp":
        return lambda y, u, force, parameters: jax.vmap(
            lambda state: jax.jvp(
                lambda z: dynamics(z, u, force, parameters), (state,), (y_direction,)
            )
        )(y[None, :] + jnp.arange(4)[:, None] * 1e-3)
    if workload.startswith("rollout_loss_"):
        argnums = {
            "rollout_loss_state": 0,
            "rollout_loss_input": 1,
            "rollout_loss_parameters": 3,
            "rollout_loss_mixed": (0, 1, 3),
            "rollout_loss_checkpointed": (0, 1, 3),
        }[workload]

        def loss(y, u, force, parameters):
            final = rollout_final_position(
                model_with_system_parameters(model, parameters),
                y,
                u,
                force,
                rollout_steps=rollout_steps,
                rollout_dt=rollout_dt,
                use_custom_jvp=use_custom_jvp,
                checkpoint=workload == "rollout_loss_checkpointed",
            )
            return jnp.mean(final**2)

        return jax.value_and_grad(loss, argnums=argnums)

    if workload == "primal":
        return dynamics
    if workload == "state_jvp":
        return lambda y, u, tau_ext, parameters: jax.jvp(
            lambda y_value: dynamics(y_value, u, tau_ext, parameters),
            (y,),
            (y_direction,),
        )
    if workload == "state_jacobian_matvec":
        if use_custom_jvp:
            return lambda y, u, tau_ext, parameters: (
                dynamics(y, u, tau_ext, parameters),
                model.forward_dynamics_state_jacobian(
                    t,
                    y,
                    (u, tau_ext),
                )
                @ y_direction,
            )
        return lambda y, u, tau_ext, parameters: (
            dynamics(y, u, tau_ext, parameters),
            jax.jacfwd(lambda y_value: dynamics(y_value, u, tau_ext, parameters))(y)
            @ y_direction,
        )
    if workload == "state_jacobian":
        return lambda y, u, tau_ext, parameters: jax.jacfwd(
            lambda y_value: dynamics(y_value, u, tau_ext, parameters)
        )(y)
    if workload == "all_jacobians":
        return lambda y, u, tau_ext, parameters: jax.jacfwd(
            dynamics,
            argnums=(0, 1, 2),
        )(y, u, tau_ext, parameters)
    if workload == "rollout_actuation_jvp":
        return lambda y, u, tau_ext, parameters: jax.jvp(
            lambda u_value: rollout(model, y, u_value, tau_ext),
            (u,),
            (u_direction,),
        )
    if workload == "rollout_actuation_jacobian":
        return lambda y, u, tau_ext, parameters: jax.jacfwd(
            lambda u_value: rollout(model, y, u_value, tau_ext)
        )(u)
    if workload == "rollout_initial_position_jvp":
        return lambda y, u, tau_ext, parameters: jax.jvp(
            lambda q_value: rollout(
                model,
                jnp.concatenate([q_value, y[model.num_velocities :]]),
                u,
                tau_ext,
            ),
            (y[: model.num_velocities],),
            (y_direction[: model.num_velocities],),
        )
    if workload == "rollout_initial_position_jacobian":
        return lambda y, u, tau_ext, parameters: jax.jacfwd(
            lambda q_value: rollout(
                model,
                jnp.concatenate([q_value, y[model.num_velocities :]]),
                u,
                tau_ext,
            )
        )(y[: model.num_velocities])
    if workload == "rollout_parameters_jvp":
        return lambda y, u, tau_ext, parameters: jax.jvp(
            lambda parameter_values: rollout(
                model_with_system_parameters(model, parameter_values),
                y,
                u,
                tau_ext,
            ),
            (parameters,),
            (system_parameter_direction,),
        )
    if workload == "rollout_parameters_jacobian":
        return lambda y, u, tau_ext, parameters: jax.jacfwd(
            lambda parameter_values: rollout(
                model_with_system_parameters(model, parameter_values),
                y,
                u,
                tau_ext,
            )
        )(parameters)
    raise ValueError(f"Unknown workload: {workload}")
