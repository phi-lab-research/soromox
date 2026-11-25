# Soromox Benchmarking Toolkit

Soromox ships development benchmarking CLIs under `tools/benchmarks`:

- `benchmark_system_methods.py` profiles individual model routines (forward kinematics,
  dynamics, etc.) to track JIT compile and steady-state execution costs.
- `benchmark_model_based_control.py` profiles the default model-based controller
  and controller-facing transformed-dynamics paths for PCS and GVS.
- `benchmark_derivatives.py` compares autodiff and optional analytical rules for
  kinematics, dynamics, and energy across all soft-robot system families.

The benchmarks reported in the paper are maintained in
[soromox-paper-experiments](https://github.com/phi-lab-research/soromox-paper-experiments):

- [Sequential CPU rollouts (Section IVa)](https://github.com/phi-lab-research/soromox-paper-experiments/tree/main/paper_results/secIVa_benchmarking_sequential_cpu)
  compare SoRoMoX, SoRoSim, and PyElastica trajectories.
- [Batched GPU rollouts (Section IVb)](https://github.com/phi-lab-research/soromox-paper-experiments/tree/main/paper_results/secIVb_parallel_rollouts_gpu)
  measure simulation throughput across batch sizes and model sizes.

## Prerequisites

- Activate the `soromox` environment (or otherwise ensure the package is on `PYTHONPATH`).
- Install JAX (CPU or GPU build), Matplotlib, Seaborn (optional, for nicer plots),
  and any accelerator-specific drivers. GPU users should follow the
  [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Benchmarking individual system methods

The `benchmark_system_methods.py` CLI times a range of core routines for a sweep of
system sizes. Each measurement captures a cold call (compile + first execution) and
reports the median of synchronized warm calls as steady-state latency.

```bash
python tools/benchmarks/benchmark_system_methods.py \
  --device cpu \
  --systems articulated_soft_robot pendulum planar_pcs pcs \
  --segment-counts 1 3 5 7 \
  --duration 2.0 \
  --solver-dt 5e-4 \
  --warmup-duration 1 \
  --execution-repeats 20 \
  --csv benchmarks/methods.csv \
  --plot benchmarks/methods.png
```

### Key options

- `--systems`: subset of available robots (defaults to all registered systems).
- `--segment-counts`: link/segment sweep; a fresh system instance is created per value.
- `--duration`, `--solver-dt` (`--dt` alias), `--save-dt`: integration controls when benchmarking
  `rollout_to`.
- `--device`: select `cpu`, `gpu`, or JAX's default automatic device choice.
  Explicit choices are applied before JAX is imported.
- `--warmup-duration`: synchronized, unmeasured execution time used to settle
  CPU frequency scaling and runtime worker threads before collecting samples.
- `--execution-repeats`: number of synchronized calls whose median is reported
  after the cold run and optional warmup.
- `--backend`: dynamics execution policy (`auto`, `jax`, or `warp`) for GVS,
  PCS, and PlanarPCS. Use separate processes when comparing backends so their
  compilation and allocator state remain isolated.
- `--json` / `--csv`: export raw results for regression tracking.
- `--plot` / `--show-plot`: render Matplotlib summaries (compile vs. exec time).

### Interpreting the results

For each system/function pair the script prints:

1. Cold-call latency (compile + execution), synchronised via `block_until_ready()`.
2. Median warm-call latency, reflecting the steady-state cost once XLA caches
   the executable and rejecting transient scheduler outliers.
3. Derived compile time = cold − warm. Pure-Python methods show near-zero compile
   time but still track runtime cost.

The plots group results by system, highlighting how complexity evolves with segment
count for each tracked method.

### Pinning CPU benchmarks

On machines with heterogeneous CPU cores, process migration between performance
and efficiency cores can distort sub-millisecond measurements. Select the CPU
device with `--device cpu` and, where the operating system supports it, pin the
whole benchmark process to a known core. On Linux, first inspect the topology and
maximum frequencies:

```bash
lscpu -e=CPU,CORE,MAXMHZ,ONLINE
```

Then run the benchmark with `taskset`; CPU 0 is only an example and should be
replaced with a performance core identified on the benchmark host:

```bash
taskset -c 0 python tools/benchmarks/benchmark_system_methods.py \
  --device cpu \
  --systems pcs \
  --segment-counts 1 4 8 16 \
  --warmup-duration 1 \
  --execution-repeats 20
```

Affinity applies to the Python process and its JAX worker threads. JSON and CSV
rows record `device`, the requested and resolved execution backends, whether
backend selection applies to the measured method, compact `cpu_affinity`,
warmup duration, and sample count so the execution context remains auditable.
When comparing revisions, use the same affinity and warmup settings for every
run.

For `articulated_soft_robot`, the method benchmark includes both the default
articulated-body forward dynamics path and a dense Jacobian-energy forward
dynamics solve (`forward_dynamics_dense`). Comparing these two cases is useful
for tracking ABA performance against the controller-facing dense dynamics API.

## Benchmarking model-based control

`benchmark_model_based_control.py` measures the checked-out implementation of
the computed-torque and configuration-/actuation-space feedforward terms, plus
the operational-space dynamics terms used by operational controllers. Compare
revisions by running the same command in separate clean worktrees.

```bash
python tools/benchmarks/benchmark_model_based_control.py \
  --systems pcs gvs \
  --segment-counts 1 4 \
  --gauss-points 5 \
  --execution-repeats 50 \
  --json /tmp/model-based-control.json
```

Use `--methods` to select individual controller paths and `--csv` for a
tabular export. Each row reports derived compile time and synchronized warm
execution time for the default implementation on that revision.

## Benchmarking derivatives

`benchmark_derivatives.py` compares the same public operation in default autodiff
and analytical mode. Analytical mode enables optional custom rules and their AD
fallbacks. The execution backend is JAX in both modes.

```bash
python tools/benchmarks/benchmark_derivatives.py \
  --device cpu --systems pendulum planar_pcs pcs gvs \
  --segment-counts 1 4 --gauss-points 5 \
  --queries pose_q forward_dynamics_state total_energy_state \
  --representations jvp vjp jacfwd jacrev \
  --execution-repeats 10 --csv /tmp/derivatives.csv --json /tmp/derivatives.json
```

- `--systems all` covers Pendulum, ArticulatedSoftRobot, McKibbenActuatedUMArm,
  PlanarPCS, PlanarHSA, PCS, ISupport, GVS, and Helicoidal. PlanarHSA and UMArm
  use the bundled parameter assets at their fixed sizes, once per sweep.
- `--queries all` covers pose and geometric-Jacobian derivatives, dynamics terms,
  forward and inverse dynamics, all five energies, and short Euler rollouts.
  Query names specify the differentiated input, such as `pose_s`,
  `forward_dynamics_control`, or `inverse_dynamics_acceleration`.
- `--representations` selects JVPs, VJPs, or full forward/reverse Jacobians.
- `--device cpu` or `--device gpu` places model data and compiled work on that
  device. An unavailable device raises an error.
- Compilation is timed separately. Runtime is the median of synchronized calls
  after warmup. Every comparison checks finite outputs and numerical agreement;
  `--tolerance` controls the maximum scale-normalized error (default `1e-5`).

CSV/JSON rows include model size, query, representation, mode, compilation/runtime,
compiler temporary bytes, correctness differences, and environment provenance.
Speedup is autodiff runtime divided by the selected mode's runtime.

The tool consolidates the former `benchmark_derivative_paths.py` and
`pcs_custom_jvp_speed_test.py` scripts. Use `--help` for the complete query list.

## Benchmarking parallel rollouts

The Section IVb generator and publication plotter are available in
[the parallel-rollout experiment](https://github.com/phi-lab-research/soromox-paper-experiments/tree/main/paper_results/secIVb_parallel_rollouts_gpu).
Follow its README for environment setup, GPU/CPU selection, batch scaling,
solver settings, failure reports, and result provenance. Run those commands
from the experiments repository.

## Visualising existing runs

To revisit stored measurements (JSON or CSV) without re-running the benchmarks, use:

```bash
python tools/benchmarks/visualize_system_methods_results.py benchmarks/methods.json \
  --systems planar_pcs pcs \
  --functions rollout_to forward_dynamics \
  --output benchmarks/methods-focus.png
```

The helper mirrors the plotting style of `benchmark_system_methods.py` and accepts
optional filters for systems/functions. Pass `--show` to open a window interactively.

The [Section IVb publication plotter](https://github.com/phi-lab-research/soromox-paper-experiments/blob/main/paper_results/secIVb_parallel_rollouts_gpu/README.md)
recreates the paper figures from the stored measurements.

## Extending the registry

The library development CLIs share `tools/benchmarks/_benchmark_common.py`.
To add a new system:

1. Implement a factory that builds the system for a requested segment/link count.
2. Provide a context builder returning representative `q`, `qd`, `u`, `tau_ext`, and
   any auxiliary data required by the benchmarking cases.
3. Register the new entry in the shared registry so it automatically becomes available
   to every benchmarking tool.

Following this pattern keeps benchmarks for future robot models (e.g., new
articulated variants or GVS variants) consistent and easy to maintain.
