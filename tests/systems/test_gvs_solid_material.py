"""Independent regression checks for material-derived solid GVS matrices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.systems import (
    GVS,
    GVSSegment,
    IsotropicMaterialParams,
    JointSpec,
    LinearProfile,
    LinkSpec,
    StrainBasisSpec,
)

jax.config.update("jax_enable_x64", True)


def _properties(kind, dimensions):
    """Area moments from the original solid-section geometry formulas."""
    a = dimensions[..., 0]
    if kind == "circular":
        area = np.pi * a**2
        iy = iz = np.pi * a**4 / 4
    else:
        b = dimensions[..., 1]
        if kind == "rectangular":
            area, iy, iz = a * b, a * b**3 / 12, b * a**3 / 12
        else:
            area, iy, iz = np.pi * a * b, np.pi * a * b**3 / 4, np.pi * b * a**3 / 4
    return area, iy, iz


@pytest.mark.parametrize("kind", ["circular", "rectangular", "elliptical"])
@pytest.mark.parametrize("tapered", [False, True])
@pytest.mark.parametrize("order", [0, 2])
@pytest.mark.parametrize("scaled", [False, True])
def test_gvs_solid_material_matrices_at_construction_and_after_updates(
    kind, tapered, order, scaled
):
    """Preserve solid laws through projection, length scaling and material updates.

    Args:
        kind: Solid cross-section family.
        tapered: Whether dimensions vary linearly along the link.
        order: Maximum monomial degree for each strain component.
        scaled: Whether rotational basis rows are divided by link length.

    Returns:
        None after checking stiffness, damping and mass against independent formulas.
    """
    length, density = 0.3, 1200.0
    start = np.array([0.02] if kind == "circular" else [0.02, 0.06])
    end = (
        start * (np.array([1.5]) if kind == "circular" else [1.5, 0.7])
        if tapered
        else start
    )
    names = {
        "circular": ("radius",),
        "rectangular": ("height", "width"),
        "elliptical": ("semi_major", "semi_minor"),
    }[kind]
    dimensions = {
        name: LinearProfile(a, b) if tapered else float(a)
        for name, a, b in zip(names, start, end, strict=True)
    }
    young, shear, damping = 1e6, 4e5, 100.0
    robot = GVS.from_segments(
        [
            GVSSegment(
                link=getattr(LinkSpec, kind)(
                    length=length,
                    density=density,
                    young_modulus=young,
                    shear_modulus=shear,
                    material_damping_coefficient=damping,
                    reference_strain=[0, 0, 0, 1, 0, 0],
                    **dimensions,
                ),
                joint=JointSpec(type="fixed"),
                basis=StrainBasisSpec(
                    type="monomial",
                    strain_selector=[1] * 6,
                    basis_order=[order] * 6,
                ),
                num_gauss_points=5,
            )
        ],
        scale_rotational_basis_by_length=scaled,
    )
    # Independent quadrature integrates section and basis products (degree <= 8).
    points, weights = np.polynomial.legendre.leggauss(8)
    points, weights = (points + 1) / 2, weights / 2
    for updated in (False, True):
        if updated:
            start, end = 1.2 * start, 1.2 * end
            length, density = 1.3 * length, 1.1 * density
            young, shear, damping = 1.4e6, 5e5, 170.0
            section = robot.params.link.cross_section
            geometry_robot = robot.update_link_params(
                cross_section=section.replace(coefficients=1.2 * section.coefficients),
                length=jnp.array([length]),
                density=jnp.array([density]),
            )
            # Geometry updates preserve canonical matrices until material is reapplied.
            assert_allclose(
                geometry_robot.params.link.stiffness, robot.params.link.stiffness
            )
            assert_allclose(
                geometry_robot.params.link.damping, robot.params.link.damping
            )
            robot = geometry_robot.with_isotropic_material(
                IsotropicMaterialParams(
                    young_modulus=young,
                    shear_modulus=shear,
                    material_damping_coefficient=damping,
                )
            )
        area, iy, iz = _properties(kind, start + points[:, None] * (end - start))
        polar = iy + iz
        stiffness = np.stack(
            [
                shear * polar,
                young * iy,
                young * iz,
                young * area,
                shear * area,
                shear * area,
            ],
            axis=-1,
        )
        damping_diagonal = damping * np.stack(
            [polar, 3 * iy, 3 * iz, 3 * area, area, area], axis=-1
        )
        # Independent monomial basis, including the off-diagonal modal couplings.
        basis = np.stack(
            [np.kron(np.eye(6), point ** np.arange(order + 1)) for point in points]
        )
        if scaled:
            basis[:, :3] /= length
        assert_allclose(
            robot.params.link.stiffness[0],
            length * np.einsum("p,pri,pr,prj->ij", weights, basis, stiffness, basis),
            rtol=1e-12,
        )
        assert_allclose(
            robot.params.link.damping[0],
            length
            * np.einsum("p,pri,pr,prj->ij", weights, basis, damping_diagonal, basis),
            rtol=1e-12,
        )
        count = int(robot.num_integration_points[0])
        mass_points = np.asarray(robot.integration_points[0, :count])
        area, iy, iz = _properties(kind, start + mass_points[:, None] * (end - start))
        mass = density * np.stack([iy + iz, iy, iz, area, area, area], axis=-1)
        assert_allclose(
            robot.mass_matrices[0, :count],
            jax.vmap(jnp.diag)(jnp.asarray(mass)),
            rtol=1e-12,
        )
