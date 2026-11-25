"""Derivative benchmark coverage, numerical checks, and mode/device isolation."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox import custom_jvp_enabled, custom_jvp_mode
from tools.benchmarks import benchmark_derivatives as benchmark
from tools.benchmarks._derivative_systems import SYSTEMS, build_system

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize(
    "query", ["pose_q", "forward_dynamics_state", "total_energy_state"]
)
def test_model_families_agree_in_both_modes(system, query):
    model = build_system(system, 1, 5)
    function, inputs = benchmark.build_query(model, query)
    values = []
    try:
        for enabled in (False, True):
            jax.clear_caches()
            with custom_jvp_mode(enabled):
                values.append(jax.jit(benchmark.derivative(function, "jvp"))(*inputs))
        absolute, relative = benchmark.difference(*values)
        assert np.isfinite(absolute)
        assert relative < 1e-5
    finally:
        jax.clear_caches()


@pytest.mark.parametrize("representation", benchmark.REPRESENTATIONS)
def test_representations_match_known_derivatives(representation):
    def operation(x, y):
        return {"sum": x + 2 * y, "square": x**2}

    x, y = jnp.array([2.0, 3.0]), jnp.array([4.0, 5.0])
    result = jax.jit(benchmark.derivative(operation, representation))(x, y)
    if representation == "jvp":
        assert_allclose(result[1]["sum"], jnp.full(2, 0.03))
        assert_allclose(result[1]["square"], 0.02 * x)
    elif representation == "vjp":
        assert_allclose(result[1][0], 1 + 2 * x)
        assert_allclose(result[1][1], jnp.full(2, 2.0))
    else:
        assert_allclose(result["sum"][0], jnp.eye(2))
        assert_allclose(result["sum"][1], 2 * jnp.eye(2))
        assert_allclose(result["square"][0], jnp.diag(2 * x))
        assert_allclose(result["square"][1], jnp.zeros((2, 2)))


@pytest.mark.parametrize("query", benchmark.QUERIES)
def test_queries_support_all_representations(query):
    model = build_system("pendulum", 1, 5)
    function, inputs = benchmark.build_query(model, query, rollout_steps=2)
    forward = jax.jit(benchmark.derivative(function, "jacfwd"))(*inputs)
    reverse = jax.jit(benchmark.derivative(function, "jacrev"))(*inputs)
    assert benchmark.difference(forward, reverse)[1] < 1e-10
    for form in ("jvp", "vjp"):
        output = jax.jit(benchmark.derivative(function, form))(*inputs)
        assert all(np.all(np.isfinite(value)) for value in jax.tree.leaves(output))
    jax.clear_caches()


@pytest.mark.parametrize(
    "reference,candidate",
    [
        (jnp.array([jnp.nan]), jnp.array([jnp.nan])),
        (jnp.array([jnp.inf]), jnp.array([jnp.inf])),
        (jnp.ones(1), jnp.ones(2)),
        ({"a": jnp.ones(1)}, {"b": jnp.ones(1)}),
    ],
)
def test_invalid_outputs_fail_agreement(reference, candidate):
    assert benchmark.difference(reference, candidate) == (float("inf"), float("inf"))


@pytest.mark.parametrize(
    "option,value",
    [
        ("--rollout-steps", "0"),
        ("--rollout-dt", "nan"),
        ("--segment-counts", "0"),
        ("--execution-repeats", "0"),
        ("--warmup-runs", "-1"),
        ("--tolerance", "inf"),
        ("--arc-length-fraction", "1"),
    ],
)
def test_invalid_options_are_rejected(option, value):
    with pytest.raises(SystemExit):
        benchmark.parse_args([option, value])


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_run_preserves_mode_and_checks_results(initial, fail, monkeypatch):
    seen = []

    def measurement(function, inputs, repeats, warmup_runs):
        seen.append(custom_jvp_enabled())
        output = function(*inputs)
        if fail and custom_jvp_enabled():
            output = jax.tree.map(lambda value: jnp.full_like(value, jnp.nan), output)
        return {"execution_time_s": 0.1, "compile_time_s": 0.2}, output

    monkeypatch.setattr(benchmark, "measure", measurement)
    args = benchmark.parse_args(
        ["--systems", "pendulum", "--queries", "pose_q", "--representations", "jvp"]
    )
    with custom_jvp_mode(initial):
        if fail:
            with pytest.raises(RuntimeError, match="derivative disagreement"):
                benchmark.run(args)
        else:
            rows = benchmark.run(args)
            assert [row["mode"] for row in rows] == ["autodiff", "analytical"]
            assert all(row["device"] == "cpu" for row in rows)
        assert custom_jvp_enabled() is initial
    assert seen == [False, True]
    jax.clear_caches()


def test_cli_exports_both_modes_on_requested_device(tmp_path):
    path = tmp_path / "derivatives.json"
    assert (
        benchmark.main(
            [
                "--systems",
                "pendulum",
                "--device",
                "cpu",
                "--queries",
                "pose_q",
                "--representations",
                "jvp",
                "--execution-repeats",
                "1",
                "--warmup-runs",
                "0",
                "--json",
                str(path),
                "--csv",
                str(path.with_suffix(".csv")),
            ]
        )
        == 0
    )
    rows = json.loads(path.read_text())
    assert len(rows) == 2
    assert all(row["device"] == "cpu" and row["compile_time_s"] > 0 for row in rows)
    assert path.with_suffix(".csv").read_text().count("\n") == 3
