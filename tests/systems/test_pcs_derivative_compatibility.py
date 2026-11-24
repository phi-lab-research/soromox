"""Analytical PCS rules must preserve the current model and execution contracts."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.actuation.friction import ThreadlikeFriction
from soromox.systems import PCS, LinkSpec, PCSStructure

jax.config.update("jax_enable_x64", True)


def _model(kind="standard"):
    actuators = None
    if kind == "friction":
        routing = ThreadlikeRouting.linear(
            intercept=jnp.array([[0.0, 0.01, 0.0]]),
            slope=jnp.zeros((1, 3)),
            start_segment_index=(0,),
            end_segment_index=(0,),
        )
        actuators = ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.3)
        )
    model = PCS.from_links(
        [
            LinkSpec.circular(
                length=0.1,
                radius=0.02,
                density=1000.0,
                young_modulus=2e3,
                shear_modulus=1e3,
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
        ],
        structure=PCSStructure(
            num_gauss_points=3,
            strain_selector=jnp.array([False, False, True, True, False, False]),
        ),
        actuators=actuators,
        floating_base=kind == "floating",
        backend="jax",
    )
    if kind == "inertia":
        model = model.update_link_params(tip_inertia=0.01 * jnp.eye(6)[None])
    return model


def _inputs(model):
    q = jnp.full((model.num_coordinates,), 0.02)
    if model.floating_base:
        q = q.at[:4].set(jnp.array([0.0, 0.0, 0.0, 1.0]))
    v = jnp.full((model.num_velocities,), 0.01)
    y = model.pack_state(q, v)
    u = jnp.full((model.num_actuators,), 0.03)
    force = jnp.full((model.num_velocities,), 0.01)
    return y, u, force


@pytest.mark.parametrize("kind", ["floating", "inertia", "friction"])
def test_extended_physics_preserves_dynamics_derivatives(kind):
    model = _model(kind)
    assert not model._supports_analytical_dynamics()
    y, u, force = _inputs(model)
    direction = jnp.linspace(-0.01, 0.02, y.size)

    def reference(state):
        return model._forward_dynamics(0.0, state, (u, force))

    def public(state):
        return model.forward_dynamics(0.0, state, (u, force))

    expected = jax.jvp(reference, (y,), (direction,))
    actual = jax.jit(lambda state: jax.jvp(public, (state,), (direction,)))(y)
    assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
    cotangent = jnp.linspace(-0.1, 0.2, y.size)
    assert_allclose(
        jax.grad(lambda state: cotangent @ public(state))(y),
        jax.grad(lambda state: cotangent @ reference(state))(y),
        rtol=1e-7,
        atol=1e-8,
    )
    assert_allclose(
        model.forward_dynamics_state_jacobian(0.0, y, (u, force)),
        jax.jacfwd(reference)(y),
        rtol=1e-7,
        atol=1e-8,
    )
    jax.clear_caches()


def test_input_and_external_tangents_skip_state_recurrence(monkeypatch):
    model = _model()
    y, u, force = _inputs(model)

    def forbidden(*args, **kwargs):
        raise AssertionError("An input tangent must not evaluate the state recurrence")

    monkeypatch.setattr(PCS, "_dynamics_terms_with_state_pushforward", forbidden)
    monkeypatch.setattr(PCS, "_inertial_force_state_pushforward", forbidden)
    directions = (jnp.ones_like(u), jnp.ones_like(force))
    expected = jax.jvp(
        lambda control, external: model._forward_dynamics(0.0, y, (control, external)),
        (u, force),
        directions,
    )
    actual = jax.jvp(
        lambda control, external: model.forward_dynamics(0.0, y, (control, external)),
        (u, force),
        directions,
    )
    assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_forward_dynamics_validates_explicit_backend():
    model = _model()
    y, u, force = _inputs(model)
    with pytest.raises(ValueError):
        model.forward_dynamics(0.0, y, (u, force), backend="invalid")
