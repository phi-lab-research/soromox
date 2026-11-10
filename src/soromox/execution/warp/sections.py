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
    """Return the area and second moments of a centred section.

    Args:
        geometry: CrossSectionGeometry value: 0 for circular, 1 for rectangular,
            or 2 for solid elliptical sections. The caller validates this value.
        outer: Outer dimensions [m]: radius in the first slot for circles,
            height/width for rectangles, or elliptical semi-axes along y/z.
        inner: Centred cavity dimensions [m], in the same order as ``outer``.
            Use zero dimensions for solids; solid ellipses use ``outer`` only.

    Returns:
        Vector ``(Ix, Iy, Iz, area)``. The first three entries are second moments
        [m^4], with ``Ix = Iy + Iz``; the last entry is material area [m^2].
    """
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
    """Compute a rectangular effective shear area from Jourawski stress energy.

    Three-point Gauss quadrature integrates Q²/b exactly within each strip of
    constant material breadth, including both symmetric halves of the section.

    Args:
        depth: Outer dimension along the applied shear direction [m].
        breadth: Outer dimension perpendicular to that direction [m].
        inner_depth: Centred cavity dimension along the shear direction [m].
            Use zero for a solid section.
        inner_breadth: Centred cavity dimension perpendicular to the shear [m].
            Use zero for a solid section. The caller validates that both cavity
            dimensions are smaller than their outer dimensions.

    Returns:
        Effective shear area [m^2] obtained by matching the energy of the
        assumed Jourawski shear field.
    """
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
    """Build a differentiable Warp function for one rod section.

    The cache key contains geometry, law and cavity choices. Dimensions and
    elastic Poisson ratio remain runtime inputs. The caller validates finite
    positive dimensions, strictly interior cavities, a uniform rectangular
    wall, and ``-1 < nu <= 0.5`` for Cowper shear before launching.

    Args:
        geometry: Circular, rectangular or solid elliptical section family.
        law: Independent transverse-shear and torsion choices.
        hollow: Whether inner dimensions describe a centred cavity.

    Returns:
        Specialized function ``section(outer, inner, nu)`` returning
        ``(inertia, young, shear, damping)`` in angular-linear strain order.
        Multiply inertia by density, form stiffness as ``E * young + G * shear``,
        and form Kelvin–Voigt damping as ``eta * damping``. The returned
        function's docstring describes its arguments, component units and outputs.

    Raises:
        ValueError: If the geometry value or geometry/law/cavity combination
            is unsupported.
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
        """Evaluate sectional inertia and unit-material constitutive diagonals.

        Geometry, law and cavity topology are fixed by the enclosing factory.

        Args:
            outer: Outer dimensions [m]: circle radius in slot zero (slot one is
                unused), rectangle height/width, or ellipse semi-axes along y/z.
            inner: Centred cavity dimensions [m], ordered like ``outer``. Solid
                specializations use zero cavity dimensions independently of this input.
            nu: Elastic Poisson ratio, dimensionless. Cowper shear requires
                ``-1 < nu <= 0.5`` and uses the same value for stiffness and damping.
                Other shear laws are independent of this input.

        Returns:
            Tuple ``(inertia, young, shear, damping)`` of six-vectors in angular-linear
            x/y/z order. ``inertia`` is ``(Ix, Iy, Iz, A, A, A)``, with moments [m^4]
            and area [m^2]. ``young`` is ``(0, Iy, Iz, A, 0, 0)``; ``shear`` is
            ``(J, 0, 0, 0, As_y, As_z)``; ``damping`` is
            ``(J, 3*Iy, 3*Iz, 3*A, As_y, As_z)``. J has units [m^4] and the
            effective shear areas have units [m^2]. Rotational mass inertia uses
            the polar moment Ix independently of the torsion constant J.
        """
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
    """Build a batched kernel for the selected rod section law.

    Samples may span links or worlds sharing the static choices, including
    samples of tapered profiles. Allocate and retain input/output buffers before
    launching or capturing a CUDA graph. Launch the returned kernel with ``dim=N``.

    Args:
        geometry: Circular, rectangular or solid elliptical section family.
        law: Independent transverse-shear and torsion choices.
        hollow: Whether sections have a centred cavity.

    Returns:
        Specialized kernel ``evaluate(outer, inner, nu, inertia, young, shear,
        damping)`` that writes four arrays of shape ``(N, 6)``. Its docstring
        specifies every input and output buffer, component units and mutation.

    Raises:
        ValueError: If the geometry value or geometry/law/cavity combination
            is unsupported.
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
        """Write the sectional inertia and material diagonals for a batch of samples.

        Launch with ``dim=N``. Geometry and law choices come from the enclosing
        factory. Each thread owns one row in each distinct caller-owned output.

        Args:
            outer: Outer dimensions [m], shape ``(N,)`` of two-component vectors.
                Store circle radius in slot zero, rectangle height/width, or ellipse
                semi-axes along material y/z.
            inner: Cavity dimensions [m], shape ``(N,)`` in the same vector layout.
                Solid specializations are independent of these values.
            nu: Elastic Poisson ratios, dimensionless, shape ``(N,)``. Used by
                Cowper shear in both stiffness and damping.
            inertia: Output unit-density inertia diagonals, shape ``(N, 6)``:
                ``(Ix, Iy, Iz, A, A, A)``, with angular entries [m^4] and linear
                entries [m^2].
            young: Output unit-Young's-modulus diagonals, shape ``(N, 6)``:
                ``(0, Iy, Iz, A, 0, 0)``. Moments are [m^4] and area is [m^2].
            shear: Output unit-shear-modulus diagonals, shape ``(N, 6)``:
                ``(J, 0, 0, 0, As_y, As_z)``. J is [m^4]; effective areas are [m^2].
            damping: Output unit-damping diagonals, shape ``(N, 6)``:
                ``(J, 3*Iy, 3*Iz, 3*A, As_y, As_z)``. Angular entries are [m^4]
                and linear entries are [m^2].

        Returns:
            None. Overwrites ``inertia``, ``young``, ``shear`` and ``damping`` in place.
        """
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
    """Project sectional material diagonals through GVS strain bases.

    Launch over ``(num_links, max_dof, max_dof)``. Each thread integrates one
    matrix entry of ``B.T @ diag(c) @ B`` over a link's contiguous sample range.
    Links may use different quadrature counts and sectional laws. The kernel
    supports reverse-mode differentiation through its floating-point inputs.

    Args:
        offsets: Sample boundaries, shape ``(num_links + 1,)``. Link i uses
            samples ``offsets[i]:offsets[i + 1]``.
        basis: Strain bases, shape ``(num_samples, 6, max_dof)``, including
            the model's rotational length scaling and zero-padded coordinates.
            Each row maps generalized coordinates to angular strain [1/m] or
            dimensionless linear strain in angular-linear x/y/z order.
        weights: Physical quadrature weights [m], shape ``(num_samples,)``,
            including each link's length.
        young: Unit-Young's-modulus diagonals, shape ``(num_samples, 6)``:
            ``(0, Iy, Iz, A, 0, 0)``, with moments [m^4] and area [m^2].
        shear: Unit-shear-modulus diagonals, shape ``(num_samples, 6)``:
            ``(J, 0, 0, 0, As_y, As_z)``, with J [m^4] and shear areas [m^2].
        damping: Unit-damping diagonals, shape ``(num_samples, 6)``:
            ``(J, 3*Iy, 3*Iz, 3*A, As_y, As_z)``, with angular entries [m^4]
            and linear entries [m^2].
        young_operators: Output unit-Young's-modulus matrices, shape
            ``(num_links, max_dof, max_dof)``.
        shear_operators: Output unit-shear-modulus matrices, same shape.
            Form generalized stiffness as ``E * young_operators + G * shear_operators``.
        damping_operators: Output unit-damping matrices, same shape.
            Form generalized damping as ``eta * damping_operators``.

    Returns:
        None. Overwrites all three output operator arrays in place. Their units
        follow the supplied generalized-coordinate basis and sectional diagonals.
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
