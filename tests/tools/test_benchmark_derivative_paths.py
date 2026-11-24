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
    initial = benchmark.custom_jvp_enabled()
    benchmark._configure_strategy("direct_analytic")
    try:
        actual = eqx.filter_jit(functions["direct_analytic"])(model, context)
        expected = eqx.filter_jit(functions["protected_autograd"])(model, context)
        for a, b in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            assert_allclose(a, b, rtol=1e-6, atol=1e-8)
    finally:
        benchmark.set_custom_jvp_enabled(initial)
        jax.clear_caches()


@pytest.mark.parametrize("initial", [False, True])
def test_benchmark_entry_points_preserve_caller_mode(initial, monkeypatch):
    from tools.benchmarks import pcs_custom_jvp_speed_test as dynamics_benchmark

    with benchmark.custom_jvp_mode(initial):
        energy_args = benchmark.parse_args([])
        energy_args.systems = []
        monkeypatch.setattr(benchmark, "parse_args", lambda argv: energy_args)
        assert benchmark.main([]) == 0
        assert benchmark.custom_jvp_enabled() is initial

        dynamics_args = dynamics_benchmark.parse_args([])
        dynamics_args.setups = []
        assert dynamics_benchmark.run(dynamics_args) == []
        assert benchmark.custom_jvp_enabled() is initial
