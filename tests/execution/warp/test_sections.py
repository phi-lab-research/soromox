# ruff: noqa: E402
"""Native rod sections: law coverage, limits, material projection and adjoints."""

import itertools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from soromox.execution.warp.sections import (
    project_rod_material_kernel,
    rod_section_function,
    rod_section_kernel,
)
from soromox.systems.components.cross_sections import (
    CrossSectionGeometry as Geometry,
)
from soromox.systems.components.cross_sections import (
    section_properties,
)
from soromox.systems.components.rod_constitutive_laws import (
    RodConstitutiveLaw,
)
from soromox.systems.components.rod_constitutive_laws import (
    TorsionRodModel as Torsion,
)
from soromox.systems.components.rod_constitutive_laws import (
    TransverseShearRodModel as Shear,
)

jax.config.update("jax_enable_x64", True)

CASES = list(itertools.product(Geometry, Shear, Torsion, [False, True]))
VALID = [
    (geometry, RodConstitutiveLaw(shear, torsion), hollow)
    for geometry, shear, torsion, hollow in CASES
    if (
        not hollow
        or (
            geometry != Geometry.ELLIPTICAL
            and shear == Shear.EFFECTIVE_AREA
            and torsion == Torsion.SAINT_VENANT
        )
    )
    and (geometry != Geometry.ELLIPTICAL or shear == Shear.UNIFORM)
]


def device():
    return "cuda:0" if os.environ.get("SOROMOX_TEST_CUDA") == "1" else "cpu"


def samples(geometry, hollow):
    """Asymmetric shapes, taper, auxetic materials and near limiting cavities."""
    scale = np.array([1.0, 0.8, 0.7, 0.6, 0.9, 1.1])
    outer = scale[:, None] * np.array([[0.04, 0.07]])
    inner = np.zeros_like(outer)
    if hollow:
        # Thin walls, moderate walls and near cavity closure, for nonsquare tubes.
        fraction = np.array([1e-4, 0.05, 0.3, 0.7, 0.999, 1 - 1e-7])
        if geometry == Geometry.CIRCULAR:
            inner[:, 0] = outer[:, 0] * (1 - fraction)
        else:
            inner = outer - fraction[:, None] * np.min(outer, axis=1)[:, None]
    nu = np.array([-0.5, 0.0, 0.3, 0.45, 0.49, 0.5])
    return outer, inner, nu


def reference(geometry, law, hollow, outer, inner, nu):
    dims = outer[:, :1] if geometry == Geometry.CIRCULAR else outer
    cavity = inner[:, :1] if geometry == Geometry.CIRCULAR else inner
    cavity = cavity if hollow else None
    polar, iy, iz, area = section_properties(geometry, dims, cavity)
    inertia = jnp.stack((polar, iy, iz, area, area, area), axis=-1)
    return (inertia, *law.material_diagonals(geometry, dims, cavity, poisson_ratio=nu))


def buffers(outer, inner, nu, *, grad=False):
    inputs = [
        wp.array(a, dtype=dtype, device=device(), requires_grad=grad)
        for a, dtype in zip(
            (outer, inner, nu), (wp.vec2d, wp.vec2d, wp.float64), strict=True
        )
    ]
    outputs = [
        wp.empty((len(nu), 6), dtype=wp.float64, device=device(), requires_grad=grad)
        for _ in range(4)
    ]
    return inputs, outputs


@pytest.mark.parametrize("geometry,law,hollow", VALID)
def test_section_values_and_parameter_updates(geometry, law, hollow):
    outer, inner, nu = samples(geometry, hollow)
    kernel = rod_section_kernel(geometry, law, hollow=hollow)
    assert kernel is rod_section_kernel(geometry, law, hollow=hollow)
    inputs, outputs = buffers(outer, inner, nu)
    for scale in (1.0, 1.3):
        inputs[0].assign(outer * scale)
        inputs[1].assign(inner * scale)
        for out in outputs:
            out.fill_(np.nan)
        wp.launch(kernel, dim=len(nu), inputs=inputs, outputs=outputs, device=device())
        expected = reference(geometry, law, hollow, outer * scale, inner * scale, nu)
        for actual, target in zip(outputs, expected, strict=True):
            np.testing.assert_allclose(actual.numpy(), target, rtol=2e-11, atol=1e-20)


@pytest.mark.parametrize("geometry,shear,torsion,hollow", CASES)
def test_static_choices_validate_before_compilation(geometry, shear, torsion, hollow):
    law = RodConstitutiveLaw(shear, torsion)
    if (geometry, law, hollow) in VALID:
        assert rod_section_function(geometry, law, hollow=hollow) is not None
    else:
        with pytest.raises(ValueError):
            rod_section_kernel(geometry, law, hollow=hollow)


@pytest.mark.parametrize("geometry,law,hollow", VALID)
def test_section_adjoints_match_jax(geometry, law, hollow):
    outer, inner, nu = samples(geometry, hollow)
    # Derivatives away from geometric endpoints; compare all output channels.
    outer, inner, nu = outer[2:3], inner[2:3], nu[2:3]
    inputs, outputs = buffers(outer, inner, nu, grad=True)
    kernel = rod_section_kernel(geometry, law, hollow=hollow)
    rng = np.random.default_rng(22)
    seeds = [rng.normal(size=(1, 6)) for _ in outputs]
    with wp.Tape() as tape:
        wp.launch(kernel, dim=1, inputs=inputs, outputs=outputs, device=device())
    tape.backward(
        grads={
            out: wp.array(seed, dtype=wp.float64, device=device())
            for out, seed in zip(outputs, seeds, strict=True)
        }
    )
    _, pullback = jax.vjp(
        lambda a, b, n: reference(geometry, law, hollow, a, b, n),
        jnp.asarray(outer),
        jnp.asarray(inner),
        jnp.asarray(nu),
    )
    expected = pullback(tuple(jnp.asarray(s) for s in seeds))
    for actual, target in zip(inputs, expected, strict=True):
        np.testing.assert_allclose(actual.grad.numpy(), target, rtol=2e-10, atol=2e-14)


def test_projection_values_and_adjoints():
    """Mixed sample counts, dense bases, padded DOFs and physical weights."""
    rng = np.random.default_rng(17)
    offsets = np.array([0, 5, 8], np.int32)
    basis = rng.normal(size=(8, 6, 7))
    basis[5:, :, 4:] = 0.0
    weights = rng.uniform(0.001, 0.1, size=8)
    diagonals = [rng.uniform(0.01, 0.8, size=(8, 6)) for _ in range(3)]
    arrays = [
        wp.array(a, dtype=wp.float64, device=device(), requires_grad=True)
        for a in [basis, weights, *diagonals]
    ]
    offset_array = wp.array(offsets, dtype=wp.int32, device=device())
    outputs = [
        wp.empty((2, 7, 7), dtype=wp.float64, device=device(), requires_grad=True)
        for _ in range(3)
    ]
    with wp.Tape() as tape:
        wp.launch(
            project_rod_material_kernel,
            dim=(2, 7, 7),
            inputs=[offset_array, *arrays],
            outputs=outputs,
            device=device(),
        )

    def ref(basis, weights, *diagonals):
        return tuple(
            jnp.stack(
                [
                    jnp.einsum(
                        "p,pri,pr,prj->ij",
                        weights[a:b],
                        basis[a:b],
                        diagonal[a:b],
                        basis[a:b],
                    )
                    for a, b in zip(offsets[:-1], offsets[1:], strict=True)
                ]
            )
            for diagonal in diagonals
        )

    expected, pullback = jax.vjp(
        ref, *[jnp.asarray(a) for a in [basis, weights, *diagonals]]
    )
    seeds = [rng.normal(size=(2, 7, 7)) for _ in outputs]
    tape.backward(
        grads={
            out: wp.array(seed, dtype=wp.float64, device=device())
            for out, seed in zip(outputs, seeds, strict=True)
        }
    )
    for actual, target in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(actual.numpy(), target, rtol=2e-13, atol=2e-15)
    for actual, target in zip(
        arrays, pullback(tuple(jnp.asarray(s) for s in seeds)), strict=True
    ):
        np.testing.assert_allclose(actual.grad.numpy(), target, rtol=2e-12, atol=2e-14)


def test_mixed_gvs_material_projection():
    """Native sections and projection reproduce actual tapered GVS operators."""
    from soromox.systems.gvs.construction import (
        _basis_at_points,
        material_operators_from_params,
        section_dimensions_at_points,
    )
    from soromox.utils.integration import gauss_quadrature
    from tests.systems.test_gvs_hollow_links import link_spec, model

    system = model(
        link_spec("circular", tapered=True),
        link_spec("rectangular", tapered=True),
        link_spec("elliptical", wall=None),
    )
    params, structure = system.params, system.structure
    poisson = np.array([0.3, 0.2, 0.4])
    expected = material_operators_from_params(
        params, structure, poisson_ratios=jnp.asarray(poisson)
    )
    bases, weights, sectional = [], [], [[], [], []]
    offsets = [0]
    for index, segment in enumerate(structure.segments):
        points, quadrature, _ = gauss_quadrature(segment.num_gauss_points)
        basis = _basis_at_points(segment, points, system.max_dof)
        length = params.link.length[index]
        if structure.scale_rotational_basis_by_length:
            basis = basis.at[:, :3, :].divide(length)
        bases.append(np.asarray(basis))
        weights.append(np.asarray(length * quadrature))
        outer, inner = section_dimensions_at_points(
            params.link.cross_section.coefficients[index], segment.link, points
        )
        hollow = inner is not None
        outer = np.asarray(outer)
        inner = np.asarray(inner) if hollow else np.zeros_like(outer)
        if outer.shape[-1] == 1:
            outer, inner = (
                np.pad(outer, ((0, 0), (0, 1))),
                np.pad(inner, ((0, 0), (0, 1))),
            )
        inputs, outputs = buffers(outer, inner, np.full(len(points), poisson[index]))
        wp.launch(
            rod_section_kernel(
                segment.link.cross_section_geometry,
                segment.link.constitutive_law,
                hollow=hollow,
            ),
            dim=len(points),
            inputs=inputs,
            outputs=outputs,
            device=device(),
        )
        for collected, output in zip(sectional, outputs[1:], strict=True):
            collected.append(output.numpy())
        offsets.append(offsets[-1] + len(points))
    inputs = [wp.array(offsets, dtype=wp.int32, device=device())] + [
        wp.array(np.concatenate(a), dtype=wp.float64, device=device())
        for a in [bases, weights, *sectional]
    ]
    outputs = [
        wp.empty(target.shape, dtype=wp.float64, device=device()) for target in expected
    ]
    wp.launch(
        project_rod_material_kernel,
        dim=expected[0].shape,
        inputs=inputs,
        outputs=outputs,
        device=device(),
    )
    for actual, target in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(actual.numpy(), target, rtol=3e-12, atol=1e-20)


@pytest.mark.parametrize("geometry", [Geometry.CIRCULAR, Geometry.RECTANGULAR])
@pytest.mark.parametrize("aspect", [1.0, 2.0, 10.0])
def test_cavity_closure_and_thin_wall_limits(geometry, aspect):
    law = RodConstitutiveLaw(Shear.EFFECTIVE_AREA, Torsion.SAINT_VENANT)
    outer = np.array([[0.04, 0.04 * aspect]])
    nu = np.array([0.3])
    solid_inputs, solid_outputs = buffers(outer, np.zeros_like(outer), nu)
    wp.launch(
        rod_section_kernel(geometry, law),
        dim=1,
        inputs=solid_inputs,
        outputs=solid_outputs,
        device=device(),
    )
    if geometry == Geometry.CIRCULAR:
        inner = np.array([[outer[0, 0] * 1e-8, 0]])
    else:
        inner = outer - outer.min() * (1 - 1e-10)
    inputs, outputs = buffers(outer, inner, nu)
    kernel = rod_section_kernel(geometry, law, hollow=True)
    wp.launch(kernel, dim=1, inputs=inputs, outputs=outputs, device=device())
    for actual, target in zip(outputs, solid_outputs, strict=True):
        np.testing.assert_allclose(
            actual.numpy(), target.numpy(), rtol=4e-9, atol=1e-20
        )
    # Independent limiting shear factors for circles, exact 5/6 for solid rectangles.
    factor = 6 * 1.3 / (7 + 6 * 0.3) if geometry == Geometry.CIRCULAR else 5 / 6
    np.testing.assert_allclose(
        solid_outputs[2].numpy()[0, 4:] / solid_outputs[0].numpy()[0, 3],
        factor,
        rtol=1e-13,
    )
    if geometry == Geometry.CIRCULAR:
        inner = np.array([[outer[0, 0] * (1 - 1e-6), 0]])
        inputs[1].assign(inner)
        wp.launch(kernel, dim=1, inputs=inputs, outputs=outputs, device=device())
        np.testing.assert_allclose(
            outputs[2].numpy()[0, 4:] / outputs[0].numpy()[0, 3],
            2 * 1.3 / (4 + 3 * 0.3),
            rtol=1e-11,
        )
    else:
        thickness = outer.min() * 1e-4
        inputs[1].assign(outer - 2 * thickness)
        wp.launch(kernel, dim=1, inputs=inputs, outputs=outputs, device=device())
        mid = outer[0] - thickness
        bredt = 4 * thickness * np.prod(mid) ** 2 / (2 * mid.sum())
        np.testing.assert_allclose(outputs[2].numpy()[0, 0] / bredt, 1, rtol=5e-8)


@pytest.mark.skipif(
    os.environ.get("SOROMOX_TEST_CUDA") != "1", reason="requires CUDA graph capture"
)
def test_section_cuda_graph_reuses_runtime_inputs():
    law = RodConstitutiveLaw(Shear.EFFECTIVE_AREA, Torsion.SAINT_VENANT)
    outer, inner, nu = samples(Geometry.RECTANGULAR, True)
    inputs, outputs = buffers(outer, inner, nu)
    kernel = rod_section_kernel(Geometry.RECTANGULAR, law, hollow=True)
    wp.launch(kernel, dim=len(nu), inputs=inputs, outputs=outputs, device=device())
    with wp.ScopedCapture(device=device()) as capture:
        wp.launch(kernel, dim=len(nu), inputs=inputs, outputs=outputs, device=device())
    inputs[0].assign(outer * 1.2)
    inputs[1].assign(inner * 1.2)
    wp.capture_launch(capture.graph)
    expected = reference(Geometry.RECTANGULAR, law, True, outer * 1.2, inner * 1.2, nu)
    for actual, target in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(actual.numpy(), target, rtol=2e-11, atol=1e-20)
