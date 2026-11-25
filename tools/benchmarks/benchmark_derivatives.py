#!/usr/bin/env python3
"""Compare autodiff and optional analytical rules for system derivatives.

Example:
    python tools/benchmarks/benchmark_derivatives.py --device cpu \
        --systems pcs --segment-counts 1 16 \
        --queries forward_dynamics_state --representations jvp vjp jacfwd jacrev

Analytical mode enables the package's custom rules, including their AD fallbacks.
Both modes differentiate the same public methods with the JAX execution backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from soromox import custom_jvp_mode
from soromox.systems import SoftRobot
from tools.benchmarks._benchmark_common import (
    benchmark_environment_metadata,
    block_until_ready,
    system_gauss_point_metadata,
)
from tools.benchmarks._derivative_systems import (
    FIXED_SIZE_SYSTEMS,
    SYSTEMS,
    build_system,
)

QUERIES = {
    "pose_q": ("forward_kinematics", ("q",)),
    "pose_s": ("forward_kinematics", ("s",)),
    "jacobian_q": ("jacobian", ("q",)),
    "jacobian_s": ("jacobian", ("s",)),
    "dynamics_terms_state": ("dynamics_terms", ("q", "v")),
    "forward_dynamics_state": ("forward_dynamics", ("y",)),
    "forward_dynamics_control": ("forward_dynamics", ("u",)),
    "forward_dynamics_all": ("forward_dynamics", ("y", "u", "external")),
    "inverse_dynamics_state": ("inverse_dynamics", ("q", "v")),
    "inverse_dynamics_acceleration": ("inverse_dynamics", ("a",)),
    "inverse_dynamics_all": ("inverse_dynamics", ("q", "v", "a")),
    **{
        f"{kind}_energy_state": (
            f"{kind}_energy",
            ("q", "v") if kind in ("kinetic", "total") else ("q",),
        )
        for kind in ("gravitational", "elastic", "kinetic", "potential", "total")
    },
    "rollout_state": ("rollout", ("y",)),
    "rollout_control": ("rollout", ("u",)),
}
REPRESENTATIONS = ("jvp", "vjp", "jacfwd", "jacrev")


def build_query(
    model: SoftRobot,
    query: str,
    *,
    arc_length_fraction: float = 0.37,
    rollout_steps: int = 8,
    rollout_dt: float = 1e-4,
) -> tuple[Callable, tuple[jax.Array, ...]]:
    """Bind a public operation while leaving the selected inputs differentiable.

    Args:
        model: Fixed-base soft robot.
        query: Operation/input selection in ``QUERIES``.
        arc_length_fraction: Interior location for scalar kinematics queries.
        rollout_steps: Number of fixed-base Euler steps for rollout queries.
        rollout_dt: Euler step duration.

    Returns:
        Callable and tuple of its array inputs. Other inputs remain constant.
    """
    q = jnp.linspace(-0.02, 0.03, model.num_coordinates)
    v = jnp.linspace(-0.03, 0.04, model.num_velocities)
    if hasattr(model, "total_length"):
        length = model.total_length
    elif hasattr(model, "segment_lengths"):
        length = jnp.sum(model.segment_lengths)
    else:
        length = jnp.sum(model.L)
    context = {
        "q": q,
        "v": v,
        "y": model.pack_state(q, v),
        "u": jnp.zeros(model.num_actuators),
        "external": jnp.zeros_like(v),
        "a": jnp.full_like(v, 0.01),
        "s": jnp.asarray(arc_length_fraction) * length,
    }
    method, names = QUERIES[query]

    def evaluate(*values):
        inputs = context | dict(zip(names, values, strict=True))
        if method in ("forward_kinematics", "jacobian"):
            return getattr(model, method)(inputs["q"], inputs["s"])
        if method == "dynamics_terms":
            return model.dynamics_terms(inputs["q"], inputs["v"])
        if method == "forward_dynamics":
            return model.forward_dynamics(
                0.0, inputs["y"], (inputs["u"], inputs["external"])
            )
        if method == "inverse_dynamics":
            return model.inverse_dynamics(inputs["q"], inputs["v"], inputs["a"])
        if method == "rollout":

            def advance(state, _):
                return state + rollout_dt * model.forward_dynamics(
                    0.0, state, (inputs["u"], inputs["external"])
                ), None

            return jax.lax.scan(advance, inputs["y"], None, length=rollout_steps)[0]
        energy = getattr(model, method)
        return (
            energy(inputs["q"], inputs["v"])
            if method in ("kinetic_energy", "total_energy")
            else energy(inputs["q"])
        )

    return evaluate, tuple(context[name] for name in names)


def derivative(function: Callable, representation: str) -> Callable:
    """Apply one derivative representation to all inputs of an operation.

    Args:
        function: Array-input operation with an array or PyTree output.
        representation: ``jvp``, ``vjp``, ``jacfwd``, or ``jacrev``.

    Returns:
        Derivative callable. Products also return the primal output.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown derivative representation: {representation}")

    def evaluate(*inputs):
        if representation == "jvp":
            directions = tuple(jnp.full_like(value, 0.01) for value in inputs)
            return jax.jvp(function, inputs, directions)
        if representation == "vjp":
            value, pullback = jax.vjp(function, *inputs)
            return value, pullback(jax.tree.map(jnp.ones_like, value))
        return getattr(jax, representation)(
            function, argnums=tuple(range(len(inputs)))
        )(*inputs)

    return evaluate


def measure(
    function: Callable, inputs: tuple, repeats: int, warmup_runs: int
) -> tuple[dict, Any]:
    """Compile once and time synchronized executions separately.

    Args:
        function: Callable to compile for the input shapes and selected mode.
        inputs: Array inputs on the selected device.
        repeats: Positive number of timed executions.
        warmup_runs: Nonnegative number of unmeasured calls.

    Returns:
        Timing/buffer metadata and final output. Runtime is the median call time.
    """
    start = time.perf_counter()
    executable = jax.jit(function).lower(*inputs).compile()
    compile_time = time.perf_counter() - start
    for _ in range(warmup_runs):
        block_until_ready(executable(*inputs))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = executable(*inputs)
        block_until_ready(output)
        times.append(time.perf_counter() - start)
    memory = executable.memory_analysis()
    return {
        "compile_time_s": compile_time,
        "execution_time_s": statistics.median(times),
        "execution_stdev_s": statistics.pstdev(times),
        "temporary_bytes": getattr(memory, "temp_size_in_bytes", None),
    }, output


def difference(reference: Any, candidate: Any) -> tuple[float, float]:
    """Compare result trees, treating shape mismatches and nonfinite values as errors.

    Args:
        reference: Autodiff output tree.
        candidate: Output from optional analytical rules.

    Returns:
        Maximum absolute and scale-normalized differences. Invalid trees return infinities.
    """
    refs, structure = jax.tree.flatten(reference)
    values, other = jax.tree.flatten(candidate)
    if structure != other:
        return float("inf"), float("inf")
    absolute = relative = 0.0
    for ref, value in zip(refs, values, strict=True):
        ref, value = np.asarray(ref), np.asarray(value)
        if (
            ref.shape != value.shape
            or not np.all(np.isfinite(ref))
            or not np.all(np.isfinite(value))
        ):
            return float("inf"), float("inf")
        error = float(np.max(np.abs(ref - value), initial=0.0))
        scale = float(np.max(np.abs(ref), initial=0.0))
        absolute = max(absolute, error)
        relative = max(relative, error / max(1.0, scale))
    return absolute, relative


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse workload and device selections.

    Args:
        argv: Argument strings, or ``None`` for the process command line.

    Returns:
        Validated options with ``all`` selections expanded.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--systems", nargs="+", choices=(*SYSTEMS, "all"), default=["pcs"]
    )
    parser.add_argument("--segment-counts", nargs="+", type=int, default=[1])
    parser.add_argument("--gauss-points", type=int, default=5)
    parser.add_argument(
        "--queries",
        nargs="+",
        choices=(*QUERIES, "all"),
        default=["pose_q", "forward_dynamics_state", "total_energy_state"],
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
    )
    parser.add_argument("--execution-repeats", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--arc-length-fraction", type=float, default=0.37)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--rollout-dt", type=float, default=1e-4)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if (
        min(
            *args.segment_counts,
            args.gauss_points,
            args.execution_repeats,
            args.rollout_steps,
        )
        < 1
    ):
        parser.error(
            "Sizes, quadrature counts, repeats, and rollout steps must be positive."
        )
    if args.warmup_runs < 0 or not 0 < args.arc_length_fraction < 1:
        parser.error(
            "Warmup count must be nonnegative and arc-length fraction must lie in (0, 1)."
        )
    if (
        not np.isfinite(args.rollout_dt)
        or args.rollout_dt <= 0
        or not np.isfinite(args.tolerance)
        or args.tolerance <= 0
    ):
        parser.error("Rollout duration and tolerance must be finite and positive.")
    args.systems = (
        list(SYSTEMS) if "all" in args.systems else list(dict.fromkeys(args.systems))
    )
    args.queries = (
        list(QUERIES) if "all" in args.queries else list(dict.fromkeys(args.queries))
    )
    return args


def run(args: argparse.Namespace) -> list[dict]:
    """Compare both modes on each selected query and device, validating every result.

    Args:
        args: Options from :func:`parse_args`.

    Returns:
        One result row per system, size, query, representation, and derivative mode.

    Raises:
        RuntimeError: If the requested device is unavailable or results fail agreement checks.
    """
    device = jax.devices(args.device)[0]
    metadata = benchmark_environment_metadata(device)
    rows = []
    with jax.default_device(device):
        for name in args.systems:
            sizes = [1] if name in FIXED_SIZE_SYSTEMS else args.segment_counts
            for size in sizes:
                model = build_system(name, size, args.gauss_points)
                gauss, _ = system_gauss_point_metadata(model)
                for query in args.queries:
                    function, inputs = build_query(
                        model,
                        query,
                        arc_length_fraction=args.arc_length_fraction,
                        rollout_steps=args.rollout_steps,
                        rollout_dt=args.rollout_dt,
                    )
                    for representation in args.representations:
                        measurements, outputs = {}, {}
                        for mode in ("autodiff", "analytical"):
                            jax.clear_caches()
                            with custom_jvp_mode(mode == "analytical"):
                                measurements[mode], outputs[mode] = measure(
                                    derivative(function, representation),
                                    inputs,
                                    args.execution_repeats,
                                    args.warmup_runs,
                                )
                        absolute, relative = difference(
                            outputs["autodiff"], outputs["analytical"]
                        )
                        if relative > args.tolerance:
                            raise RuntimeError(
                                f"{name}/{query}/{representation}: derivative disagreement ({relative:g})"
                            )
                        for mode, timing in measurements.items():
                            rows.append(
                                metadata
                                | timing
                                | {
                                    "system": name,
                                    "size_label": "fixed"
                                    if name in FIXED_SIZE_SYSTEMS
                                    else "links/segments",
                                    "size": getattr(
                                        model,
                                        "num_segments",
                                        getattr(model, "num_links", size),
                                    ),
                                    "coordinates": model.num_coordinates,
                                    "velocities": model.num_velocities,
                                    "gauss_points": gauss,
                                    "query": query,
                                    "representation": representation,
                                    "mode": mode,
                                    "device": device.platform,
                                    "speedup": measurements["autodiff"][
                                        "execution_time_s"
                                    ]
                                    / timing["execution_time_s"],
                                    "max_abs_diff": absolute,
                                    "max_scaled_diff": relative,
                                    "execution_repeats": args.execution_repeats,
                                    "warmup_runs": args.warmup_runs,
                                    "arc_length_fraction": args.arc_length_fraction,
                                    "rollout_steps": args.rollout_steps
                                    if query.startswith("rollout")
                                    else None,
                                    "rollout_dt": args.rollout_dt
                                    if query.startswith("rollout")
                                    else None,
                                }
                            )
                        print(
                            f"{name} size={rows[-1]['size']} {query} {representation}: "
                            f"analytical speedup {rows[-1]['speedup']:.2f}x",
                            flush=True,
                        )
                jax.clear_caches()
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write requested result files.

    Args:
        argv: CLI arguments, or ``None`` for the process command line.

    Returns:
        Zero after successful validation and export.
    """
    args = parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    rows = run(args)
    for path in (args.csv, args.json):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
    if args.csv and rows:
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
