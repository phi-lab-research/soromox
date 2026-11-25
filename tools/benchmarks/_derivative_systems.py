"""Model factories for the differentiation benchmark."""

from pathlib import Path

import jax.numpy as jnp

from soromox.systems import (
    Helicoidal,
    HelicoidalBackboneParams,
    ISupport,
    ISupportParams,
    ISupportStructure,
    McKibbenActuatedUMArm,
    PlanarHSA,
    PlanarHSAParams,
    PlanarHSAStructure,
    SoftRobot,
)
from tools.benchmarks._benchmark_common import (
    _gvs_segment,
    _pcs_factory,
    build_system_with_gauss_points,
    get_system_registry,
)

SYSTEMS = (
    "pendulum",
    "articulated_soft_robot",
    "mckibben_umarm",
    "planar_pcs",
    "planar_hsa",
    "pcs",
    "isupport",
    "gvs",
    "helicoidal",
)
FIXED_SIZE_SYSTEMS = frozenset({"mckibben_umarm", "planar_hsa"})
ASSETS = Path(__file__).resolve().parents[2] / "assets" / "robot_parameters"


def build_system(name: str, size: int, gauss_points: int) -> SoftRobot:
    """Construct a fixed-base model with differentiable execution.

    Args:
        name: Name in ``SYSTEMS``.
        size: Link/segment count for scalable models. Asset models retain their size.
        gauss_points: Quadrature nodes for continuum models; GVS requires at least five.

    Returns:
        Model using the JAX execution backend where configurable.

    Raises:
        ValueError: If the system name or quadrature count is unsupported.
    """
    registry = get_system_registry()
    if name in registry:
        config = registry[name]
        if config.supports_gauss_points and gauss_points < config.min_gauss_points:
            raise ValueError(
                f"{name} requires at least {config.min_gauss_points} Gauss points"
            )
        return build_system_with_gauss_points(
            config,
            size,
            gauss_points if config.supports_gauss_points else None,
            backend="jax",
        )
    if name == "mckibben_umarm":
        return McKibbenActuatedUMArm.from_cached_parameters(
            ASSETS / "mckibben_umarm" / "reference_parameters.npz"
        )
    if name == "planar_hsa":
        return PlanarHSA(
            params=PlanarHSAParams.from_npz(ASSETS / "planar_hsa" / "fpu_control.npz"),
            structure=PlanarHSAStructure(),
        )
    if name == "isupport":
        base = _pcs_factory(size, gauss_points, backend="jax")
        params = ISupportParams(
            link=base.params.link,
            gravity=jnp.array([0.0, 0.0, -9.81]),
            chamber_inner_radius=jnp.full(size, 0.002),
            chamber_outer_radius=jnp.full(size, 0.004),
            chamber_distance=jnp.full(size, 0.01),
            chamber_azimuth_angles=jnp.tile(2 * jnp.pi * jnp.arange(3) / 3, (size, 1)),
        )
        return ISupport(
            params,
            structure=ISupportStructure(
                rigid_segment_selector=(False,) * size,
                num_gauss_points=gauss_points,
            ),
            backend="jax",
        )
    if name == "helicoidal":
        if gauss_points < 5:
            raise ValueError("helicoidal requires at least five Gauss points")
        law = HelicoidalBackboneParams.from_compliance(
            length=0.25,
            compliance=jnp.array([5.0, 2.0, 2.0, 0.01, 0.02, 0.02]),
            compliance_gradient=jnp.array([0.0, 3.0, 3.0, 0.0, 0.0, 0.0]),
            damping_time_constant=0.02,
        )
        return Helicoidal.from_segments(
            [_gvs_segment(0, gauss_points) for _ in range(size)],
            mass_per_length=jnp.full(size, 0.2),
            rotational_inertia_per_length=jnp.tile(
                jnp.array([2e-4, 1e-4, 1e-4]), (size, 1)
            ),
            tip_mass=jnp.full(size, 0.04),
            constitutive_params=dict.fromkeys(range(size), law),
            backend="jax",
        )
    raise ValueError(f"Unknown system: {name}")
