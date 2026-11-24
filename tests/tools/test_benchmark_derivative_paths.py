"""Checks that direct energy benchmark references track the model hook API."""

import equinox as eqx
import jax
import pytest
from numpy.testing import assert_allclose

from tools.benchmarks import benchmark_derivative_paths as benchmark
from tools.benchmarks._benchmark_common import (
    build_system_with_gauss_points,
    get_system_registry,
)


@pytest.mark.parametrize("system_name", ["pcs", "planar_pcs", "gvs"])
@pytest.mark.parametrize("case", ["kinetic_energy_gradient", "total_energy_gradient"])
def test_direct_energy_reference_matches_protected_autodiff(system_name, case):
    config = get_system_registry()[system_name]
    model = build_system_with_gauss_points(
        config, size=1, gauss_points=5, backend="jax"
    )
    context = config.build_context(model)
    functions = benchmark._case_registry()[case]
    benchmark._configure_strategy("direct_analytic")
    try:
        actual = eqx.filter_jit(functions["direct_analytic"])(model, context)
        expected = eqx.filter_jit(functions["protected_autograd"])(model, context)
        for a, b in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            assert_allclose(a, b, rtol=1e-6, atol=1e-8)
    finally:
        benchmark.set_custom_jvp_enabled(True)
        jax.clear_caches()
