"""Composition and work-sharing contracts for analytical dynamics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.actuation import EffortModel, ThreadlikeImpedance
from soromox.systems.pcs import _derivatives
from tools.benchmarks import pcs_custom_jvp_speed_test as benchmark

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def case():
    model = benchmark.build_model("threadlike", 1, 3)
    return model, benchmark.build_inputs(model)


def assert_tree(actual, expected, rtol=1e-6, atol=1e-8):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_allclose(a, b, rtol=rtol, atol=atol)


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
def test_composed_workloads_match_reference(case, workload):
    model, inputs = case

    def make(enabled):
        return jax.jit(
            benchmark.build_workload(
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
    assert_tree(make(True)(*args), make(False)(*args))
    jax.clear_caches()


def test_jvp_vjp_duality_and_hessian(case):
    model, inputs = case

    def public(y):
        return model.forward_dynamics(0.0, y, (inputs.u, inputs.tau_ext))

    def reference(y):
        return model._forward_dynamics(0.0, y, (inputs.u, inputs.tau_ext))

    tangent = inputs.y_direction
    cotangent = jnp.flip(tangent)
    _, action = jax.jvp(public, (inputs.y,), (tangent,))
    _, pullback = jax.vjp(public, inputs.y)
    assert_allclose(cotangent @ action, tangent @ pullback(cotangent)[0], rtol=1e-8)

    def hvp(fn):
        return jax.jvp(
            jax.grad(lambda y: 1e-5 * cotangent @ fn(y)), (inputs.y,), (tangent,)
        )[1]

    assert_allclose(hvp(public), hvp(reference), rtol=1e-6, atol=1e-7)
    epsilon = 1e-5
    finite_difference = (
        reference(inputs.y + epsilon * tangent)
        - reference(inputs.y - epsilon * tangent)
    ) / (2 * epsilon)
    assert_allclose(action, finite_difference, rtol=1e-5, atol=1e-6)
    jax.clear_caches()


def test_terms_and_inverse_dynamics_compose(case):
    model, inputs = case
    passive = ThreadlikeImpedance(
        routing=model.actuators[0].transmission.routing,
        stiffness=jnp.array([10.0, 20.0]),
        damping=jnp.array([0.1, 0.2]),
        rest_length=jnp.array([0.09, 0.1]),
    )
    model = eqx.tree_at(lambda m: m.passive_elements, model, (passive,))
    q, v = jnp.split(inputs.y, 2)
    acceleration = jnp.array([0.03, -0.02])
    direction = jnp.array([0.01, 0.02])
    assert_tree(
        jax.jvp(model.dynamics_terms, (q, v), (direction, direction)),
        jax.jvp(model._assemble_dynamics_terms, (q, v), (direction, direction)),
    )

    def reference(q, v, a):
        return _derivatives.inverse_dynamics_reference(model, q, v, a)

    assert_tree(
        jax.jvp(
            model.inverse_dynamics,
            (q, v, acceleration),
            (direction, direction, direction),
        ),
        jax.jvp(reference, (q, v, acceleration), (direction, direction, direction)),
    )
    assert_tree(
        jax.jacrev(model.inverse_dynamics, argnums=(0, 1, 2))(q, v, acceleration),
        jax.jacrev(reference, argnums=(0, 1, 2))(q, v, acceleration),
    )
    effort = model.inverse_dynamics(q, v, acceleration)
    result = model.forward_dynamics(0.0, inputs.y, (jnp.zeros_like(inputs.u), effort))
    assert_allclose(result[q.size :], acceleration, rtol=1e-8, atol=1e-9)
    # Explicit environment axes and vmap must agree for the public term API.
    assert_tree(
        model.dynamics_terms(jnp.stack((q, q)), jnp.stack((v, v))),
        jax.vmap(model.dynamics_terms)(jnp.stack((q, q)), jnp.stack((v, v))),
    )
    jax.clear_caches()


def test_acceleration_only_inverse_dynamics_skips_state_actions(case, monkeypatch):
    model, inputs = case
    q, v = jnp.split(inputs.y, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("inactive state recurrence evaluated")

    monkeypatch.setattr(
        type(model), "_dynamics_terms_with_state_pushforward", forbidden
    )
    monkeypatch.setattr(type(model), "inverse_dynamics_state_pushforward", forbidden)
    result = jax.jvp(
        lambda a: model.inverse_dynamics(q, v, a), (v,), (jnp.ones_like(v),)
    )[1]
    assert_allclose(result, model._assemble_dynamics_terms(q, v)[0] @ jnp.ones_like(v))


@pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="CPU LAPACK factorization inspection"
)
@pytest.mark.parametrize("workload", ["mixed_jvp", "state_directions"])
def test_compiled_primal_and_tangents_share_one_factorization(case, workload):
    model, inputs = case
    fn = benchmark.build_workload(
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


def test_velocity_only_terms_skip_configuration_recurrence(case, monkeypatch):
    model, inputs = case
    q, v = jnp.split(inputs.y, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("inactive configuration recurrence evaluated")

    monkeypatch.setattr(
        type(model), "_dynamics_terms_with_state_pushforward", forbidden
    )
    actual = jax.jvp(lambda v: model.dynamics_terms(q, v), (v,), (jnp.ones_like(v),))
    expected = jax.jvp(
        lambda v: model._assemble_dynamics_terms(q, v), (v,), (jnp.ones_like(v),)
    )
    assert_tree(actual, expected)


@pytest.mark.parametrize("implicit", [False, True])
def test_diffrax_step_gradients_compose(case, implicit):
    import diffrax
    import optimistix

    model, inputs = case
    solver = (
        diffrax.ImplicitEuler(root_finder=optimistix.Newton(rtol=1e-8, atol=1e-10))
        if implicit
        else diffrax.Euler()
    )

    def integrate(control, analytical):
        field = model.forward_dynamics if analytical else model._forward_dynamics
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: field(t, y, (args, inputs.tau_ext))),
            solver,
            t0=0.0,
            t1=3e-4,
            dt0=1e-4,
            y0=inputs.y,
            args=control,
            adjoint=diffrax.DirectAdjoint(),
            max_steps=8,
        )
        return jnp.mean(solution.ys[-1] ** 2)

    def evaluate(analytical):
        def loss(control):
            return integrate(control, analytical)

        return (
            jax.jit(jax.value_and_grad(loss))(inputs.u),
            jax.jit(lambda u: jax.jvp(loss, (u,), (inputs.u_direction,)))(inputs.u),
        )

    assert_tree(evaluate(True), evaluate(False), rtol=2e-6)
    jax.clear_caches()


class NonlinearEffort(EffortModel):
    def effort(self, control, coordinate, velocity):
        return control**2 + coordinate + velocity**2


def test_custom_effort_uses_local_fallback(case, monkeypatch):
    model, inputs = case
    model = eqx.tree_at(
        lambda m: m.actuators[0]._effort_model, model, NonlinearEffort()
    )
    assert model._supports_analytical_inertia()
    assert not model._supports_analytical_dynamics()
    calls = []
    original = type(model)._dynamics_terms_with_state_pushforward

    def observed(self, *args):
        calls.append(True)
        return original(self, *args)

    monkeypatch.setattr(type(model), "_dynamics_terms_with_state_pushforward", observed)

    def public(y, u):
        return model.forward_dynamics(0.0, y, (u, inputs.tau_ext))

    def reference(y, u):
        return model._forward_dynamics(0.0, y, (u, inputs.tau_ext))

    primals = inputs.y, inputs.u
    directions = inputs.y_direction, inputs.u_direction
    assert_tree(
        jax.jvp(public, primals, directions), jax.jvp(reference, primals, directions)
    )
    assert calls  # The component fallback must retain analytical inertia derivatives.
    assert_tree(
        jax.jacrev(public, argnums=(0, 1))(*primals),
        jax.jacrev(reference, argnums=(0, 1))(*primals),
    )
    jax.clear_caches()


def test_inverse_dynamics_model_parameter_gradients(case):
    model, inputs = case
    q, v = jnp.split(inputs.y, 2)
    acceleration = jnp.array([0.03, -0.02])

    def evaluate(parameters, analytical):
        updated = benchmark.model_with_system_parameters(model, parameters)
        if analytical:
            return updated.inverse_dynamics(q, v, acceleration)
        return _derivatives.inverse_dynamics_reference(updated, q, v, acceleration)

    for transform in (jax.jacfwd, jax.jacrev):
        actual = jax.jit(transform(lambda p: evaluate(p, True)))(
            inputs.system_parameters
        )
        expected = jax.jit(transform(lambda p: evaluate(p, False)))(
            inputs.system_parameters
        )
        assert_tree(actual, expected)
    jax.clear_caches()


def test_large_state_platform_paths(case, monkeypatch):
    model, inputs = case
    monkeypatch.setattr(_derivatives, "FUSED_STATE_MAX_VELOCITIES", 0)

    def public(y):
        return model.forward_dynamics(0.0, y, (inputs.u, inputs.tau_ext))

    def reference(y):
        return model._forward_dynamics(0.0, y, (inputs.u, inputs.tau_ext))

    for transform in (jax.jacfwd, jax.jacrev):
        assert_tree(
            jax.jit(transform(public))(inputs.y),
            jax.jit(transform(reference))(inputs.y),
        )
    q, v = jnp.split(inputs.y, 2)

    def required(q):
        return model.inverse_dynamics(q, v, v)

    def raw(q):
        return _derivatives.inverse_dynamics_reference(model, q, v, v)

    assert_tree(jax.jit(jax.jacfwd(required))(q), jax.jit(jax.jacfwd(raw))(q))
    assert_tree(jax.jit(jax.jacrev(required))(q), jax.jit(jax.jacrev(raw))(q))
    jax.clear_caches()
