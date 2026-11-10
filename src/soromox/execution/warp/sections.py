"""Differentiable FP64 sectional mechanics for Warp-native rod models.

Specialize on validated geometry and constitutive choices once, then evaluate
runtime dimensions and Poisson ratios in caller-owned buffers. These kernels
belong to parameter preparation, before the kinematics/dynamics time loop.
They use the same unit-material convention as RodConstitutiveLaw.material_diagonals.
"""

from functools import cache

import warp as wp

from soromox.systems.components.cross_sections import CrossSectionGeometry
from soromox.systems.components.rod_constitutive_laws import (
    RodConstitutiveLaw,
    TorsionRodModel,
    TransverseShearRodModel,
)

__all__ = ["rod_section_function", "rod_section_kernel", "project_rod_material_kernel"]

# Define this locally: the dynamics spatial helpers disable Warp autodiff.
Vec6d = wp.types.vector(length=6, dtype=wp.float64)


@wp.func
def _section_properties(geometry: int, outer: wp.vec2d, inner: wp.vec2d) -> wp.vec4d:
    """Return (polar moment, Iy, Iz, area) for centred material boundaries."""
    a, b = outer[0], outer[1]
    c, d = inner[0], inner[1]
    area = wp.float64(0.0)
    iy = wp.float64(0.0)
    iz = wp.float64(0.0)
    pi = wp.float64(3.141592653589793)
    if geometry == 0:
        area = pi * (a * a - c * c)
        iy = pi * ((a * a * a * a) - (c * c * c * c)) / wp.float64(4.0)
        iz = iy
    elif geometry == 1:
        area = a * b - c * d
        iy = (a * (b * b * b) - c * (d * d * d)) / wp.float64(12.0)
        iz = (b * (a * a * a) - d * (c * c * c)) / wp.float64(12.0)
    else:
        area = pi * a * b
        iy = pi * a * (b * b * b) / wp.float64(4.0)
        iz = pi * (a * a * a) * b / wp.float64(4.0)
    return wp.vec4d(iy + iz, iy, iz, area)


@wp.func
def _rectangular_shear_area(
    depth: wp.float64,
    breadth: wp.float64,
    inner_depth: wp.float64,
    inner_breadth: wp.float64,
) -> wp.float64:
    """Integrate Jourawski Q²/b exactly with three Gauss points per strip."""
    half = depth / wp.float64(2.0)
    inner_half = inner_depth / wp.float64(2.0)
    strip = half - inner_half
    central_breadth = breadth - inner_breadth
    energy = wp.float64(0.0)
    for i in range(3):
        node = wp.float64(0.0)
        weight = wp.float64(8.0) / wp.float64(9.0)
        if i != 1:
            node = wp.float64(i - 1) * wp.sqrt(wp.float64(3.0) / wp.float64(5.0))
            weight = wp.float64(5.0) / wp.float64(9.0)
        central_u = inner_half * (wp.float64(1.0) + node) / wp.float64(2.0)
        central_q = (
            central_breadth * (inner_half - central_u) * (inner_half + central_u)
            + breadth * strip * (half + inner_half)
        ) / wp.float64(2.0)
        outer_u = inner_half + strip * (wp.float64(1.0) + node) / wp.float64(2.0)
        outer_q = breadth * (half - outer_u) * (half + outer_u) / wp.float64(2.0)
        energy += weight * (
            inner_half * central_q * central_q / central_breadth
            + strip * outer_q * outer_q / breadth
        )
    moment = (
        breadth * (depth * depth * depth)
        - inner_breadth * (inner_depth * inner_depth * inner_depth)
    ) / wp.float64(12.0)
    return moment * moment / energy


@cache
def rod_section_function(
    geometry: CrossSectionGeometry,
    law: RodConstitutiveLaw = RodConstitutiveLaw(),
    *,
    hollow: bool = False,
) -> wp.Function:
    """Build a Warp function returning unit-material diagonals for one section.

    Args:
        geometry: Circular, rectangular or solid elliptical section.
        law: Same independent shear/torsion choices as the GVS construction API.
        hollow: Whether inner dimensions describe a cavity. Hollow ellipses and
            unsupported law combinations are rejected before compilation.

    Returns:
        wp.Function: A Warp function ``f(outer: vec2d, inner: vec2d, nu: float64)`` returning
            four six-vectors ``(inertia, young, shear, damping)``. Multiply inertia
            by density, form stiffness as ``E * young + G * shear`` and damping as
            ``eta * damping``. Order is angular then linear (x, y, z).
            Inertia uses the polar area moment, independently of torsion choice.

            Dimensions are radius (second slot ignored), rectangle height/width,
            or ellipse semi-axes along y/z. Solid functions ignore ``inner``.
            Hollow rectangles require a centred uniform wall. Callers validate
            finite positive dimensions, strictly interior cavities, and
            ``-1 < nu <= 0.5`` for Cowper shear before launching. ``nu`` is the
            elastic Poisson ratio, also for damping; unused laws ignore it.

            The function supports Warp reverse-mode differentiation. Its cache key
            contains only static choices, never dimensions or material values.
    """
    geometry = CrossSectionGeometry(geometry)
    law.validate_geometry(geometry, hollow)
    if hollow and geometry == CrossSectionGeometry.ELLIPTICAL:
        raise ValueError("Hollow elliptical sections are unsupported.")
    geometry_id = int(geometry)
    effective = law.transverse_shear == TransverseShearRodModel.EFFECTIVE_AREA
    saint_venant = law.torsion == TorsionRodModel.SAINT_VENANT

    @wp.func
    def section(outer: wp.vec2d, inner: wp.vec2d, nu: wp.float64):
        cavity = wp.vec2d(wp.float64(0.0))
        if wp.static(hollow):
            cavity = inner
        props = _section_properties(wp.static(geometry_id), outer, cavity)
        polar, iy, iz, area = props[0], props[1], props[2], props[3]
        torsion = polar
        if wp.static(saint_venant and geometry_id == 1):
            thickness = wp.min(outer[0], outer[1]) / wp.float64(2.0)
            if wp.static(hollow):
                thickness = (outer[0] - cavity[0]) / wp.float64(2.0)
            mid_h, mid_w = outer[0] - thickness, outer[1] - thickness
            perimeter = wp.float64(2.0) * (mid_h + mid_w)
            mid_area = mid_h * mid_w
            torsion = wp.float64(4.0) * thickness * (
                mid_area * mid_area
            ) / perimeter + perimeter * (
                thickness * thickness * thickness
            ) / wp.float64(3.0)
        elif wp.static(saint_venant and geometry_id == 2):
            a, b = outer[0], outer[1]
            torsion = (
                wp.float64(3.141592653589793)
                * (a * a * a)
                * (b * b * b)
                / (a * a + b * b)
            )
        sy, sz = area, area
        if wp.static(effective and geometry_id == 0):
            ratio2 = (cavity[0] / outer[0]) * (cavity[0] / outer[0])
            ratio_term = (wp.float64(1.0) + ratio2) * (wp.float64(1.0) + ratio2)
            factor = (
                wp.float64(6.0)
                * (wp.float64(1.0) + nu)
                * ratio_term
                / (
                    (wp.float64(7.0) + wp.float64(6.0) * nu) * ratio_term
                    + (wp.float64(20.0) + wp.float64(12.0) * nu) * ratio2
                )
            )
            sy, sz = area * factor, area * factor
        elif wp.static(effective and geometry_id == 1):
            sy = _rectangular_shear_area(outer[0], outer[1], cavity[0], cavity[1])
            sz = _rectangular_shear_area(outer[1], outer[0], cavity[1], cavity[0])
        zero = wp.float64(0.0)
        inertia = Vec6d(polar, iy, iz, area, area, area)
        young = Vec6d(zero, iy, iz, area, zero, zero)
        shear = Vec6d(torsion, zero, zero, zero, sy, sz)
        damping = Vec6d(
            torsion,
            wp.float64(3.0) * iy,
            wp.float64(3.0) * iz,
            wp.float64(3.0) * area,
            sy,
            sz,
        )
        return inertia, young, shear, damping

    return section


@cache
def rod_section_kernel(
    geometry: CrossSectionGeometry,
    law: RodConstitutiveLaw = RodConstitutiveLaw(),
    *,
    hollow: bool = False,
) -> wp.Kernel:
    """Build a batched kernel for :func:`rod_section_function`.

    Args:
        geometry: Section geometry.
        law: Rod constitutive choices, validated by the function factory.
        hollow: Whether sections have a cavity.

    Returns:
        wp.Kernel: Kernel with inputs ``outer, inner`` (``wp.array[wp.vec2d]``), ``nu``
            (``wp.array[wp.float64]``) and outputs ``inertia, young, shear, damping``
            (each ``wp.array2d[wp.float64]`` of shape ``(N, 6)``). Launch with
            ``dim=N``; all inputs have N entries. Caller owns distinct output
            buffers. Solids ignore inner values. Samples may span any links or
            worlds sharing the static choices, including tapered profiles.
            No allocation or device transfer occurs during a launch.
    """
    section = rod_section_function(geometry, law, hollow=hollow)

    @wp.kernel
    def evaluate(
        outer: wp.array[wp.vec2d],
        inner: wp.array[wp.vec2d],
        nu: wp.array[wp.float64],
        inertia: wp.array2d[wp.float64],
        young: wp.array2d[wp.float64],
        shear: wp.array2d[wp.float64],
        damping: wp.array2d[wp.float64],
    ):
        sample = wp.tid()
        m, e, g, d = section(outer[sample], inner[sample], nu[sample])
        for row in range(6):
            inertia[sample, row] = m[row]
            young[sample, row] = e[row]
            shear[sample, row] = g[row]
            damping[sample, row] = d[row]

    return evaluate


@wp.kernel
def project_rod_material_kernel(
    offsets: wp.array[wp.int32],
    basis: wp.array3d[wp.float64],
    weights: wp.array[wp.float64],
    young: wp.array2d[wp.float64],
    shear: wp.array2d[wp.float64],
    damping: wp.array2d[wp.float64],
    young_operators: wp.array3d[wp.float64],
    shear_operators: wp.array3d[wp.float64],
    damping_operators: wp.array3d[wp.float64],
):
    """Project sectional diagonals through GVS strain bases, with autodiff.

    Launch over ``(num_links, max_dof, max_dof)``. Each thread writes one full
    matrix entry, integrating ``B.T @ diag(c) @ B`` without atomics or scratch.
    Samples for a link must occupy a contiguous range in the packed arrays.
    Mixed quadrature sizes and laws are supported by preparing each group's
    sectional output views before projection. No output initialization is needed.

    Args:
        offsets: INT32 sample boundaries, length num_links + 1.
        basis: FP64 shape (num_samples, 6, max_dof), zero padded, including any
            rotational length scaling selected by the model.
        weights: FP64 physical quadrature weights, including link length.
        young: Unit Young's-modulus diagonals, shape (num_samples, 6).
        shear: Unit shear-modulus diagonals, same shape.
        damping: Unit damping diagonals, same shape.
        young_operators: Output shape (num_links, max_dof, max_dof).
        shear_operators: Output with the same shape.
        damping_operators: Output with the same shape.
    """
    link, row, column = wp.tid()
    e, g, d = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    for sample in range(offsets[link], offsets[link + 1]):
        for strain in range(6):
            factor = (
                weights[sample]
                * basis[sample, strain, row]
                * basis[sample, strain, column]
            )
            e += factor * young[sample, strain]
            g += factor * shear[sample, strain]
            d += factor * damping[sample, strain]
    young_operators[link, row, column] = e
    shear_operators[link, row, column] = g
    damping_operators[link, row, column] = d
