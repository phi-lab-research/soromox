from __future__ import annotations

import jax
import jax.numpy as jnp
import pcs_derivative_cases as cases
import pytest
from numpy.testing import assert_allclose

from soromox.autodiff import custom_jvp_mode
from soromox.systems import PCS

jax.config.update("jax_enable_x64", True)


def _assert_trees_allclose(reference, candidate) -> None:
    reference_leaves, reference_tree = jax.tree_util.tree_flatten(reference)
    candidate_leaves, candidate_tree = jax.tree_util.tree_flatten(candidate)

    assert candidate_tree == reference_tree
    for reference_leaf, candidate_leaf in zip(
        reference_leaves,
        candidate_leaves,
        strict=True,
    ):
        assert_allclose(candidate_leaf, reference_leaf, rtol=1.0e-5, atol=1.0e-8)


def _build_workload(model: PCS, workload: str, *, use_custom_jvp: bool):
    inputs = cases.build_inputs(model)
    workload_fn = cases.build_workload(
        model,
        workload,
        inputs.y_direction,
        inputs.u_direction,
        inputs.system_parameter_direction,
        rollout_steps=3,
        rollout_dt=1.0e-4,
        use_custom_jvp=use_custom_jvp,
    )
    args = (
        inputs.y,
        inputs.u,
        inputs.tau_ext,
        inputs.system_parameters,
    )
    return inputs, workload_fn, args


def test_full_dynamics_jacobians_match_direct_autodiff() -> None:
    """The reference bypasses the optional derivative dispatch."""
    model = cases.build_model("threadlike", num_segments=2, gauss_points=5)
    _, reference_fn, inputs = _build_workload(
        model, "all_jacobians", use_custom_jvp=False
    )
    _, candidate_fn, _ = _build_workload(model, "all_jacobians", use_custom_jvp=True)

    reference = reference_fn(*inputs)
    with custom_jvp_mode(True):
        candidate = candidate_fn(*inputs)

    _assert_trees_allclose(reference, candidate)


@pytest.mark.usefixtures("analytical_derivatives")
def test_custom_jvp_input_only_tangent_matches_direct_jax_autodiff() -> None:
    """Cover the custom rule's no-state-tangent inertia-matrix branch."""
    model = cases.build_model("threadlike", num_segments=2, gauss_points=5)
    inputs = cases.build_inputs(model)
    y = inputs.y
    u = inputs.u
    tau_ext = inputs.tau_ext
    t = jnp.array(0.0)

    reference = jax.jacfwd(
        lambda u_value: model._forward_dynamics(t, y, (u_value, tau_ext))
    )(u)
    candidate = jax.jacfwd(
        lambda u_value: model.forward_dynamics(
            t,
            y,
            (u_value, tau_ext),
        )
    )(u)

    assert_allclose(candidate, reference, rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize("setup", ["full", "bending_extension"])
def test_fused_dynamics_state_pushforward_matches_direct_jax_autodiff(
    setup: str,
) -> None:
    model = cases.build_model(setup, num_segments=2, gauss_points=5)
    inputs = cases.build_inputs(model)
    q, qd = jnp.split(inputs.y, 2)
    q_direction, qd_direction = jnp.split(inputs.y_direction, 2)

    expected_primal, expected_direction = jax.jvp(
        model.dynamics_terms,
        (q, qd),
        (q_direction, qd_direction),
    )
    actual = model._dynamics_terms_with_state_pushforward(
        q,
        qd,
        q_direction,
        qd_direction,
    )
    actual_primal = actual[:3]
    actual_direction = actual[3:]

    _assert_trees_allclose(expected_primal, actual_primal)
    _assert_trees_allclose(expected_direction, actual_direction)


@pytest.mark.parametrize("num_segments", [2, 16])
@pytest.mark.usefixtures("analytical_derivatives")
def test_directional_jvp_matches_full_analytical_jacobian_matvec(
    num_segments: int,
) -> None:
    """Cover short and scan-based analytical state-JVP recurrences."""
    model = cases.build_model(
        "threadlike",
        num_segments=num_segments,
        gauss_points=5,
    )
    _, directional_fn, inputs = _build_workload(
        model,
        "state_jvp",
        use_custom_jvp=True,
    )
    _, full_jacobian_fn, _ = _build_workload(
        model,
        "state_jacobian_matvec",
        use_custom_jvp=True,
    )
    _, autodiff_full_jacobian_fn, _ = _build_workload(
        model,
        "state_jacobian_matvec",
        use_custom_jvp=False,
    )

    directional = directional_fn(*inputs)
    full_jacobian = full_jacobian_fn(*inputs)
    autodiff_full_jacobian = autodiff_full_jacobian_fn(*inputs)

    _assert_trees_allclose(directional, full_jacobian)
    _assert_trees_allclose(directional, autodiff_full_jacobian)


@pytest.mark.usefixtures("analytical_derivatives")
def test_large_dense_directional_jvp_matches_direct_jax_autodiff() -> None:
    """Cover the contracted inverse-dynamics route above the fused cutoff."""
    model = cases.build_model("full", num_segments=16, gauss_points=5)
    _, reference_fn, inputs = _build_workload(
        model,
        "state_jvp",
        use_custom_jvp=False,
    )
    _, candidate_fn, _ = _build_workload(
        model,
        "state_jvp",
        use_custom_jvp=True,
    )

    reference = reference_fn(*inputs)
    candidate = candidate_fn(*inputs)

    _assert_trees_allclose(reference, candidate)


@pytest.mark.parametrize(
    "actuation_args",
    [
        pytest.param(None, id="none"),
        pytest.param((None,), id="default-input"),
    ],
)
@pytest.mark.usefixtures("analytical_derivatives")
def test_custom_jvp_state_tangent_accepts_optional_inputs(
    actuation_args: tuple | None,
) -> None:
    model = cases.build_model("axial", num_segments=1, gauss_points=5)
    y = cases.build_inputs(model).y
    t = jnp.array(0.0)

    reference = jax.jacfwd(
        lambda y_value: model._forward_dynamics(t, y_value, actuation_args)
    )(y)
    candidate = jax.jacfwd(
        lambda y_value: model.forward_dynamics(
            t,
            y_value,
            actuation_args,
        )
    )(y)

    assert_allclose(candidate, reference, rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize(
    "workload",
    [
        "rollout_actuation_jvp",
        "rollout_actuation_jacobian",
        "rollout_initial_position_jvp",
        "rollout_initial_position_jacobian",
        "rollout_parameters_jvp",
        "rollout_parameters_jacobian",
    ],
)
@pytest.mark.usefixtures("analytical_derivatives")
def test_rollout_derivative_workloads_match_direct_jax_autodiff(workload: str) -> None:
    model = cases.build_model("axial", num_segments=1, gauss_points=5)
    _, reference_fn, inputs = _build_workload(
        model,
        workload,
        use_custom_jvp=False,
    )
    _, candidate_fn, _ = _build_workload(
        model,
        workload,
        use_custom_jvp=True,
    )

    reference = reference_fn(*inputs)
    candidate = candidate_fn(*inputs)

    _assert_trees_allclose(reference, candidate)
    derivative = reference[1] if workload.endswith("_jvp") else reference
    assert jnp.linalg.norm(derivative) > 0.0


@pytest.fixture
def derivative_case():
    model = cases.build_model("threadlike", 1, 3)
    return model, cases.build_inputs(model)


@pytest.mark.parametrize(
    "workload",
    [
        "state_vjp",
        "input_vjp",
        "mixed_jvp",
        "mixed_vjp",
        "state_jacrev",
        "state_directions",
        "batched_state_jvp",
        "rollout_loss_state",
        "rollout_loss_input",
        "rollout_loss_parameters",
        "rollout_loss_mixed",
        "rollout_loss_checkpointed",
    ],
)
@pytest.mark.usefixtures("analytical_derivatives")
def test_composed_workloads_match_reference(derivative_case, workload):
    model, inputs = derivative_case

    def make(enabled):
        return jax.jit(
            cases.build_workload(
                model,
                workload,
                inputs.y_direction,
                inputs.u_direction,
                inputs.system_parameter_direction,
                rollout_steps=3,
                rollout_dt=1e-4,
                use_custom_jvp=enabled,
            )
        )

    args = (inputs.y, inputs.u, inputs.tau_ext, inputs.system_parameters)
    _assert_trees_allclose(make(True)(*args), make(False)(*args))
    jax.clear_caches()


@pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="CPU LAPACK factorization inspection"
)
@pytest.mark.parametrize("workload", ["mixed_jvp", "state_directions"])
@pytest.mark.usefixtures("analytical_derivatives")
def test_compiled_primal_and_tangents_share_one_factorization(
    derivative_case, workload
):
    model, inputs = derivative_case
    fn = cases.build_workload(
        model,
        workload,
        inputs.y_direction,
        inputs.u_direction,
        inputs.system_parameter_direction,
        rollout_steps=3,
        rollout_dt=1e-4,
        use_custom_jvp=True,
    )
    program = (
        jax.jit(fn)
        .lower(inputs.y, inputs.u, inputs.tau_ext, inputs.system_parameters)
        .compiler_ir("hlo")
        .as_hlo_text()
    )
    assert program.count('custom_call_target="lapack_dpotrf_ffi"') == 1
