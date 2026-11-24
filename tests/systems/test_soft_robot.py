"""Unit tests for shared :class:`soromox.systems.SoftRobot` contracts."""

import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import (
    gvs_params_from_segments,
    pcs_params,
    planar_pcs_params,
)

from soromox import custom_jvp_mode
from soromox.systems import (
    GVS,
    PCS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    PCSStructure,
    PlanarPCS,
    PlanarPCSStructure,
    StrainBasisSpec,
    _dynamics,
)

jax.config.update("jax_enable_x64", True)


def _planar_robot(*, floating_base: bool) -> PlanarPCS:
    params = planar_pcs_params(
        length=jnp.array([0.1]),
        radius=jnp.array([0.01]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1.0e5]),
        shear_modulus=jnp.array([4.0e4]),
        damping_matrix=jnp.eye(3),
        gravity=jnp.array([0.0, -9.81]),
    )
    return PlanarPCS(
        params=params,
        structure=PlanarPCSStructure(num_gauss_points=3),
        floating_base=floating_base,
        backend="jax",
    )


def _spatial_robot(*, floating_base: bool) -> PCS:
    params = pcs_params(
        length=jnp.array([0.1]),
        radius=jnp.array([0.01]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1.0e5]),
        shear_modulus=jnp.array([4.0e4]),
        damping_matrix=jnp.eye(6),
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    return PCS(
        params=params,
        structure=PCSStructure(num_gauss_points=3),
        floating_base=floating_base,
        backend="jax",
    )


@pytest.mark.parametrize(
    ("factory", "base_coordinates", "base_velocities"),
    [(_planar_robot, 3, 3), (_spatial_robot, 7, 6)],
)
def test_dimensions_distinguish_storage_velocity_and_internal_coordinates(
    factory, base_coordinates: int, base_velocities: int
) -> None:
    fixed = factory(floating_base=False)
    floating = factory(floating_base=True)

    assert fixed.num_base_coordinates == 0
    assert fixed.num_base_velocities == 0
    assert fixed.num_coordinates == fixed.num_internal_dofs
    assert fixed.num_velocities == fixed.num_internal_dofs
    assert floating.num_base_coordinates == base_coordinates
    assert floating.num_base_velocities == base_velocities
    assert floating.num_coordinates == base_coordinates + floating.num_internal_dofs
    assert floating.num_velocities == base_velocities + floating.num_internal_dofs
    assert floating.state_size == (
        floating.num_coordinates
        + floating.num_velocities
        + floating.num_auxiliary_states
    )
    assert not hasattr(floating, "num_dofs")


def test_pack_and_split_helpers_broadcast_leading_axes() -> None:
    robot = _planar_robot(floating_base=True)
    q_internal = jnp.stack(
        [
            jnp.zeros((robot.num_internal_dofs,)),
            jnp.full((robot.num_internal_dofs,), 0.1),
        ]
    )
    q = robot.pack_configuration(q_internal, base_pose=jnp.array([0.2, 0.3, -0.1]))
    v = robot.pack_velocity(
        jnp.zeros((robot.num_internal_dofs,)),
        base_velocity=jnp.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]]),
    )
    y = robot.pack_state(q, v)

    q_base, q_internal_actual = robot.split_configuration(q)
    v_base, qd_internal_actual = robot.split_velocity(v)
    q_actual, v_actual, auxiliary = robot.split_state(y)
    assert q_base is not None
    assert v_base is not None
    assert q.shape == (2, robot.num_coordinates)
    assert v.shape == (2, robot.num_velocities)
    assert y.shape == (2, robot.state_size)
    assert_allclose(q_base, jnp.broadcast_to(jnp.array([0.2, 0.3, -0.1]), (2, 3)))
    assert_allclose(q_internal_actual, q_internal)
    assert_allclose(qd_internal_actual, 0.0)
    assert_allclose(q_actual, q)
    assert_allclose(v_actual, v)
    assert auxiliary.shape == (2, 0)


def test_pack_helpers_reject_wrong_base_usage_and_dimensions() -> None:
    fixed = _planar_robot(floating_base=False)
    floating = _planar_robot(floating_base=True)
    q_internal = jnp.zeros((fixed.num_internal_dofs,))

    with pytest.raises(ValueError, match="base_pose is required"):
        floating.pack_configuration(q_internal)
    with pytest.raises(ValueError, match="only valid for a floating-base robot"):
        fixed.pack_configuration(q_internal, base_pose=jnp.zeros((3,)))
    with pytest.raises(ValueError, match="trailing dimension"):
        floating.pack_configuration(q_internal, base_pose=jnp.zeros((4,)))
    with pytest.raises(ValueError, match="only valid for a floating-base robot"):
        fixed.pack_velocity(q_internal, base_velocity=jnp.zeros((3,)))


def test_spatial_normalization_validation_and_trace_safe_fallback() -> None:
    robot = _spatial_robot(floating_base=True)
    q_internal = jnp.zeros((robot.num_internal_dofs,))
    q = robot.pack_configuration(
        q_internal,
        base_pose=jnp.array([-0.2, 0.1, 0.3, 2.0, 0.4, -0.5, 0.6]),
    )

    assert_allclose(jnp.linalg.norm(q[:4]), 1.0)
    assert_allclose(robot.normalize_configuration(-q)[:4], -q[:4])
    with pytest.raises(ValueError, match="nonzero norm"):
        robot.normalize_configuration(q.at[:4].set(0.0))
    with pytest.raises(ValueError, match="finite"):
        robot.normalize_configuration(q.at[0].set(jnp.inf))

    traced = eqx.filter_jit(robot.normalize_configuration)(q.at[:4].set(0.0))
    assert_allclose(traced[:4], jnp.array([0.0, 0.0, 0.0, 1.0]))


def test_configuration_derivative_matches_retraction_tangent() -> None:
    robot = _spatial_robot(floating_base=True)
    q = robot.pack_configuration(
        jnp.linspace(-0.02, 0.03, robot.num_internal_dofs),
        base_pose=jnp.array([0.2, -0.1, 0.3, 0.9, 0.1, -0.2, 0.4]),
    )
    v = robot.pack_velocity(
        jnp.linspace(0.03, -0.02, robot.num_internal_dofs),
        base_velocity=jnp.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1]),
    )
    epsilon = 1.0e-6
    finite_difference = (robot.retract_configuration(q, epsilon * v) - q) / epsilon
    assert_allclose(
        finite_difference,
        robot.configuration_derivative(q, v),
        rtol=2.0e-6,
        atol=2.0e-7,
    )


def test_fixed_mount_update_and_public_state_projection() -> None:
    floating = _spatial_robot(floating_base=True)
    q_internal = jnp.zeros((floating.num_internal_dofs,))
    base_pose = jnp.array([0.0, 0.0, 0.0, 1.0, 0.2, -0.1, 0.3])
    q = floating.pack_configuration(q_internal, base_pose=base_pose)
    v = floating.pack_velocity(jnp.zeros((floating.num_internal_dofs,)))
    y = floating.pack_state(q.at[:4].multiply(1.8), v)

    projected = floating.project_state(y)
    projected_q, projected_v, projected_auxiliary = floating.split_state(projected)
    assert_allclose(jnp.linalg.norm(projected_q[:4]), 1.0)
    assert_allclose(projected_q[4:], q[4:])
    assert_allclose(projected_v, v)
    assert projected_auxiliary.shape == (0,)
    assert_allclose(floating.base_transform_from_configuration(q)[:3, 3], base_pose[4:])

    fixed = _spatial_robot(floating_base=False).with_fixed_base_pose(base_pose)
    assert_allclose(fixed.fixed_base_pose, base_pose)
    assert_allclose(
        fixed.base_transform_from_configuration(q_internal)[:3, 3], base_pose[4:]
    )
    with pytest.raises(ValueError, match="has no fixed base pose"):
        floating.with_fixed_base_pose(base_pose)


def _gvs_robot(*, floating_base: bool) -> GVS:
    params, structure = gvs_params_from_segments(
        [
            GVSSegment(
                link=LinkSpec.circular(
                    length=0.1,
                    radius=0.01,
                    density=1000.0,
                    young_modulus=1e5,
                    shear_modulus=4e4,
                    reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ),
                joint=JointSpec(type="fixed"),
                basis=StrainBasisSpec(
                    type="monomial", strain_selector=[1] * 6, basis_order=[0] * 6
                ),
                num_gauss_points=5,
            )
        ],
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    return GVS(
        params=params, structure=structure, floating_base=floating_base, backend="jax"
    )


@pytest.mark.parametrize("factory", [_planar_robot, _spatial_robot, _gvs_robot])
@pytest.mark.parametrize("floating_base", [False, True])
def test_required_effort_and_shared_derivative_composition(factory, floating_base):
    robot = factory(floating_base=floating_base)
    base_pose = None
    if floating_base:
        base_pose = (
            jnp.array([0.1, -0.1, 0.2])
            if robot.num_base_coordinates == 3
            else jnp.array([0.1, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0])
        )
    q = robot.pack_configuration(
        jnp.linspace(-0.02, 0.03, robot.num_internal_dofs), base_pose=base_pose
    )
    v = jnp.linspace(-0.02, 0.04, robot.num_velocities)
    a = jnp.linspace(0.01, 0.03, robot.num_velocities)
    dq, dv, da = jnp.full_like(q, 0.01), jnp.full_like(v, 0.02), jnp.full_like(a, 0.03)
    effort = robot.inverse_dynamics(q, v, a)
    assert_allclose(
        robot.passive_force(q, v), robot.elastic_force(q) + robot.damping_matrix(q) @ v
    )
    y = robot.pack_state(q, v)
    result = robot.forward_dynamics(0.0, y, (jnp.zeros(robot.num_actuators), effort))
    assert_allclose(robot.split_state(result)[1], a, rtol=1e-6, atol=1e-8)
    expected = jax.jit(
        lambda q, v, a: jax.jvp(robot._inverse_dynamics, (q, v, a), (dq, dv, da))
    )(q, v, a)
    actual = jax.jit(
        lambda q, v, a: jax.jvp(robot.inverse_dynamics, (q, v, a), (dq, dv, da))
    )(q, v, a)
    shared = eqx.filter_jit(_dynamics.inverse_dynamics_jvp)(
        (robot, q, v, a), (jax.tree.map(lambda _: None, robot), dq, dv, da)
    )
    for candidate in (actual, shared):
        for value, reference in zip(candidate, expected, strict=True):
            assert_allclose(value, reference, rtol=1e-6, atol=1e-8)
    cotangent = jnp.linspace(0.1, 0.2, robot.num_velocities)
    actual_vjp = jax.jit(
        lambda q, v, a: jax.vjp(robot.inverse_dynamics, q, v, a)[1](cotangent)
    )(q, v, a)
    expected_vjp = jax.jit(
        lambda q, v, a: jax.vjp(robot._inverse_dynamics, q, v, a)[1](cotangent)
    )(q, v, a)
    for actual, expected in zip(actual_vjp, expected_vjp, strict=True):
        assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    args = (jnp.zeros(robot.num_actuators), effort)
    dy = robot.pack_state(dq, dv)
    expected = eqx.filter_jit(
        lambda y: jax.jvp(lambda y: robot._forward_dynamics(0.0, y, args), (y,), (dy,))
    )(y)
    actual = eqx.filter_jit(_dynamics.forward_dynamics_jvp)(
        (robot, 0.0, y, args), (jax.tree.map(lambda _: None, robot), None, dy, None)
    )
    for value, reference in zip(actual, expected, strict=True):
        assert_allclose(value, reference, rtol=1e-6, atol=1e-8)
    jax.clear_caches()


@pytest.mark.parametrize("factory", [_planar_robot, _spatial_robot, _gvs_robot])
def test_disabled_pose_derivatives_skip_analytical_hooks(factory, monkeypatch):
    robot = factory(floating_base=False)
    q = jnp.linspace(-0.02, 0.03, robot.num_coordinates)
    s = jnp.array(0.05)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled analytical pose rule was called")

    monkeypatch.setattr(type(robot), "_forward_kinematics_jvp", forbidden)
    with custom_jvp_mode(False):
        actual = jax.jit(
            jax.jacfwd(lambda s: robot.forward_kinematics(q, s, backend="jax"))
        )(s)
        expected = jax.jit(
            jax.jacfwd(lambda s: robot._absolute_forward_kinematics(q, s))
        )(s)
    assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
    jax.clear_caches()


@pytest.mark.parametrize("factory", [_planar_robot, _spatial_robot, _gvs_robot])
@pytest.mark.parametrize("floating_base", [False, True])
def test_shared_energy_derivatives_compose(factory, floating_base, monkeypatch):
    from soromox.systems import _energy

    robot = factory(floating_base=floating_base)
    base_pose = None
    if floating_base:
        base_pose = (
            jnp.array([0.1, -0.1, 0.2])
            if robot.num_base_coordinates == 3
            else jnp.array([0.1, -0.1, 0.2, 0.0, 0.0, 0.0, 1.0])
        )
    q = robot.pack_configuration(
        jnp.linspace(-0.02, 0.03, robot.num_internal_dofs), base_pose=base_pose
    )
    v = jnp.linspace(-0.03, 0.04, robot.num_velocities)
    assert_allclose(
        robot.kinetic_energy(q, v),
        0.5 * v @ robot.inertia_matrix(q) @ v,
        rtol=1e-6,
        atol=1e-8,
    )
    dq, dv = jnp.full_like(q, 0.01), jnp.full_like(v, 0.02)
    original = type(robot)._analytical_energy_jvp

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled analytical energy rule was called")

    for enabled in (False, True):
        monkeypatch.setattr(
            type(robot), "_analytical_energy_jvp", original if enabled else forbidden
        )
        for kind in ("kinetic", "gravitational", "elastic", "potential", "total"):

            def fn(q, v, kind=kind):
                method = getattr(robot, f"{kind}_energy")
                return method(q, v) if kind in ("kinetic", "total") else method(q)

            def raw(q, v, kind=kind):
                return _energy.evaluate(robot, q, v, kind)

            def derivatives(function):
                return jax.jit(
                    lambda q, v: (
                        jax.jvp(function, (q, v), (dq, dv)),
                        jax.grad(function, argnums=(0, 1))(q, v),
                    )
                )(q, v)

            with custom_jvp_mode(enabled):
                actual = derivatives(fn)
                expected = derivatives(raw)
            for a, b in zip(
                jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
            ):
                assert_allclose(a, b, rtol=1e-6, atol=1e-8)
        jax.clear_caches()


@pytest.mark.parametrize("factory", [_planar_robot, _spatial_robot, _gvs_robot])
def test_energy_derivatives_avoid_execution_dispatch(factory, monkeypatch):
    """Energy values and derivatives use protected JAX equations on every backend."""
    robot = factory(floating_base=False)
    q = jnp.linspace(-0.02, 0.03, robot.num_coordinates)
    v = jnp.linspace(-0.03, 0.04, robot.num_velocities)
    dq, dv = jnp.ones_like(q), jnp.ones_like(v)

    def forbidden(*args, **kwargs):
        raise AssertionError("energy derivative entered dynamics execution dispatch")

    monkeypatch.setattr(type(robot), "dynamics_terms", forbidden)
    for enabled in (False, True):
        with custom_jvp_mode(enabled):
            fn = jax.jit(
                lambda q, v: (
                    jax.jvp(robot.total_energy, (q, v), (dq, dv)),
                    jax.grad(robot.total_energy, argnums=(0, 1))(q, v),
                    jax.grad(lambda v: robot.kinetic_energy(q, v))(v),
                )
            )
            result = fn(q, v)
        assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(result))
        jax.clear_caches()


def test_pcs_composite_energies_use_ad_gravity(monkeypatch):
    robot = _spatial_robot(floating_base=False)
    q = jnp.linspace(-0.02, 0.03, robot.num_coordinates)
    v = jnp.linspace(-0.03, 0.04, robot.num_velocities)

    def forbidden(*args, **kwargs):
        raise AssertionError("PCS energy requested analytical gravity")

    monkeypatch.setattr(type(robot), "_gravitational_force", forbidden)
    for kind in ("gravitational", "potential", "total"):

        def fn(q, public, kind=kind):
            method = getattr(robot, f"{kind}_energy" if public else f"_{kind}_energy")
            return method(q, v) if kind == "total" else method(q)

        with custom_jvp_mode(True):
            actual = jax.jit(jax.grad(lambda q: fn(q, True)))(q)
            expected = jax.jit(jax.grad(lambda q: fn(q, False)))(q)
        assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    jax.clear_caches()


def test_gvs_pose_arc_length_uses_autodiff(monkeypatch):
    robot = _gvs_robot(floating_base=False)
    q = jnp.linspace(-0.02, 0.03, robot.num_coordinates)
    s = jnp.array(0.05)

    def forbidden(*args, **kwargs):
        raise AssertionError("GVS pose requested analytical arc-length recurrence")

    monkeypatch.setattr(
        type(robot), "_forward_kinematics_and_arc_length_derivative", forbidden
    )
    with custom_jvp_mode(True):
        actual = jax.jit(
            jax.jacfwd(lambda s: robot.forward_kinematics(q, s, backend="jax"))
        )(s)
        expected = jax.jit(
            jax.jacfwd(lambda s: robot._absolute_forward_kinematics(q, s))
        )(s)
    assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    jax.clear_caches()


@pytest.mark.parametrize("factory", [_planar_robot, _spatial_robot, _gvs_robot])
@pytest.mark.parametrize("enabled", [False, True])
def test_scalar_jacobian_aliases_share_derivative_routing(
    factory, enabled, monkeypatch
):
    robot = factory(floating_base=False)
    q = jnp.linspace(-0.02, 0.03, robot.num_coordinates)
    s = jnp.array(0.05)
    original = type(robot)._jacobian_jvp
    calls = []

    def observed(self, *args, **kwargs):
        assert enabled, "disabled analytical Jacobian rule was called"
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(robot), "_jacobian_jvp", observed)
    reference = None
    for method in (robot.jacobian, robot.jacobian_inertialframe):
        calls.clear()
        with custom_jvp_mode(enabled):
            result = jax.jit(
                lambda q, method=method: (
                    jax.jvp(lambda q: method(q, s), (q,), (jnp.ones_like(q),)),
                    jax.grad(lambda q: jnp.sum(method(q, s)))(q),
                )
            )(q)
        assert bool(calls) == enabled
        if reference is None:
            reference = result
        else:
            for a, b in zip(
                jax.tree.leaves(result), jax.tree.leaves(reference), strict=True
            ):
                assert_allclose(a, b, rtol=1e-6, atol=1e-8)
        jax.clear_caches()
