# Analytical dynamics derivatives: implementation plan

Updated 2026-11-23 for PR #198, rebased onto `origin/main` at
`bfea7c961f84a531aa5ab93a66da461a76469f56`.

## Implementation follow-up

Implemented on `codex/shared-dynamics-derivatives`, based on PR #198 at
`a739725cc87e9779bcbb8180aa5227c8128b00b0`. The public interface remains ordinary
JAX transformations; there is no package-specific linearization object.

- Private `_derivatives.py` composes forward dynamics, inverse dynamics, and
  dynamics-term actions. Spatial recurrences remain shared in PCS.
- Forward dynamics factors inertia once and applies those factors to the primal
  and combined derivative right-hand side. Explicit dense helpers reuse the
  factors and solve several columns together. CPU uses the existing symmetric
  Cholesky policy; GPU retains LU for the existing generic solve semantics.
  Above 64 velocities, state derivatives use the contracted recurrence on CPU
  and a combined AD term action on accelerators. `lax.platform_dependent`
  selects at lowering, so the choice respects the actual compilation platform.
- Mixed state/model-parameter tangents differentiate one dynamics-term assembly
  and one applied-force assembly, followed by the implicit acceleration action.
  This avoids repeating the primal for separate state and parameter rules.
- Symbolically inactive state tangents skip the state recurrence. Velocity-only
  dynamics-term differentiation holds configuration fixed and uses AD pruning.
  Input-only force evaluation shares its primal and derivative. Numerical zeros
  in a packed state remain ordinary tangent values.
- Supported state recurrences can now coexist with local AD for friction, custom
  routing, or nonlinear/velocity-dependent actuator effort laws. Floating states,
  custom inertia, and PCS subclasses retain the general AD path.
- Dynamics-term derivatives use one optional hook in the shared execution layer,
  after shape/backend validation, for both JAX and Warp execution boundaries.
- `PCS.inverse_dynamics(q, qd, qdd)` returns required generalized effort including
  passive elasticity and damping. It supports ordinary JVPs/VJPs and model
  parameter differentiation. The older recurrence helpers continue to describe
  inertial effort only; their docstrings explicitly distinguish this convention.

For example, applications can use:

```python
jax.jvp(lambda y: robot.forward_dynamics(t, y, (u,)), (y,), (dy,))
jax.grad(lambda u: loss(integrate(robot, y0, u)))(u)
jax.jacfwd(robot.inverse_dynamics, argnums=(0, 1, 2))(q, qd, qdd)
```

The benchmark now includes VJPs, mixed gradients, multiple directions at a shared
state, distinct environment states, a fallback inertia model, scalar rollout
losses, and checkpointed rollout losses. It reports compilation time, median
execution time and variation, and compiler buffer estimates. Nonfinite results
fail the correctness guard.

### Measured outcome

[Raw measurements](benchmarks/pcs-shared-derivatives.csv) include both the previous
PR implementation and this implementation, each compared to an independent AD
reference. JAX 0.11.0, FP64, five quadrature points; CPU is pinned to core 7 and
GPU is an RTX 5090. GPU access requires running outside the local sandbox.
The first table uses one reduced-strain segment with threadlike actuation,
16 rollout steps, three warmups, and 10 samples of 10 calls.

| Workload | CPU before / after (ms) | GPU before / after (ms) |
|---|---:|---:|
| Mixed state/control/parameter VJP | 0.123 / 0.084 | 0.400 / 0.272 |
| Mixed rollout loss gradient | 1.601 / 0.942 | 11.912 / 8.013 |
| Same loss with checkpointing | 1.044 / 0.717 | 7.841 / 5.744 |

The mixed rollout gradient is about 1.7x faster than PR #198 on CPU and 1.5x on
GPU in this sample. It is now close to plain AD, rather than substantially
faster than AD. CPU compiler buffer estimates for that workload decrease from
1,201,200 to 776,440 bytes. GPU estimates include substantial solver workspace
(about 32 MiB in several transpose cases); they do not establish a general GPU
memory reduction. Compiler estimates are not measured peak device memory.

For a 16-segment full-strain model (96 velocities), this implementation measured
1.24x for a JVP, 1.30x for a VJP, 3.09x for a full Jacobian, and 1.71x for four
shared-state directions, relative to AD on CPU. This sample uses five warmups
and 10 samples of five calls. Small single-product timings remain variable;
the raw data includes cases below 1x. A 128-step mixed rollout takes 7.19 ms
versus AD's 7.22 ms; checkpointing takes 5.26 ms versus AD's 5.19 ms on CPU.
There is no universal speedup claim.

The large GPU model exposed the limits of the inherited CPU threshold: its
contracted recurrence took 3.35 ms for a JVP and 8.38 ms for a VJP, against AD's
1.37/1.88 ms. Fused analytical assembly improved this to 1.53/2.40 ms but still
lost to AD. The selected general term action plus shared implicit solve takes
1.35/1.89 ms, against AD's 1.38/1.95 ms in the final repeat. The raw data retains
all three experiments. The same platform policy applies to inverse-dynamics
state actions; CPU and GPU tests cover both derivative modes.

### HiJax decision

`tools/benchmarks/pcs_hijax_experiment.py` runs the actual PCS state, control,
external-force, and material-parameter equations through a HiJax primitive. It
preserves its derivative boundary under distinct-state batching and checks both
Jacobian modes and a Hessian. Run it with an optional JAX 0.12.1 environment:

```bash
PYTHONPATH=src python -m tools.benchmarks.pcs_hijax_experiment
```

This experiment carries a JAX linearization explicitly through `lin` and derives
JVP/VJP rules from it. It tests whether that boundary adds value to the shared
composition already implemented; it is not a second hand-optimized spatial
recurrence. On the confirmed CPU comparison, mixed JVP/VJP and shared directions
are approximately tied with ordinary custom JVPs, and buffer estimates are
identical. The GPU comparison is also within about 5% across the measured
workloads, with identical buffer estimates. Both device experiments pass the
higher-derivative and batching checks. [Raw HiJax results](benchmarks/pcs-hijax-comparison.json)
include the timing and memory estimates. Earlier apparent CPU improvements were
not consistent on repetition.
Keep HiJax experimental and retain one ordinary custom-JVP implementation in
production. A future explicit residual policy could still justify revisiting it.
The runtime dependency floor is unchanged.

### Validation and remaining limits

Tests cover transpose consistency, finite differences, Hessian-vector products,
JIT, environment batching, shared directions, nonlinear force fallback, and
forward/inverse dynamics consistency. Compiler inspection verifies exactly one
CPU factorization for mixed tangents and shared directions. Both explicit Euler
and implicit Euler in Diffrax pass forward/reverse differentiation comparisons
using an adjoint configuration that supports those transformations.

The isolated JAX 0.12.1 environment was upgraded with matching CUDA plugins and
CUDA library dependencies. The NVIDIA driver is available outside the sandbox.
GPU verification therefore uses actual device execution. Validation completed:

- JAX 0.11.0 CPU: 48 composition, compatibility, benchmark, and execution tests;
  30 selected PCS/actuation regressions; two final targeted checks.
- JAX 0.12.1 GPU: 24 composition/compatibility tests passed; two CPU-only
  compiler-inspection tests skipped.
- Both devices: the actual-PCS HiJax experiment passes first/second derivatives
  and distinct-state batching.
- Explicit/implicit Diffrax integration, inactive-state/configuration pruning,
  and CPU factorization-sharing checks pass.

Additional targeted tests verify inverse dynamics with passive forces and model
parameters, and platform selection for forward/inverse dynamics. The final
platform checks pass on CPU and GPU; the large CPU directional regression also
passes.

Remaining performance work is measurement-driven: larger batched GPU rollouts,
application-scale horizons, actual peak-device-memory profiling, and explicit
analytical support for floating/custom-inertia equations. Checkpointing is
exercised through standard JAX rematerialization; no custom transpose, special
checkpoint rule, or hardware-specific kernel is added without evidence that it
helps the composed workload.

## Decision and scope

Expose analytical derivatives through ordinary JAX transformations of forward
dynamics, inverse dynamics, dynamics terms, and kinematics. Users should be able
to differentiate a loss, integration step, or rollout with respect to state,
controls, external forces, and continuous model parameters. The derivative of
each component then participates in the enclosing computation's chain rule.
Do not add a public linearization object or a separate simulator gradient API.

Prioritize scalar-loss VJPs and directional JVPs alongside full Jacobians.
Trajectory optimization by gradient descent and differentiable policy training
usually need VJPs through a rollout. iLQR/DDP and many SQP/MPC implementations
need local state/control Jacobians; matrix-free solvers can use products instead.
Collocation usually assembles sparse Jacobian blocks. Standard model-free policy
gradient RL does not require simulator derivatives. The solver determines the
representation; there is no single answer for all MPC or RL applications.

[Soft Swing-up](https://arxiv.org/abs/2602.03435) evaluates collocation, DDP, and
NMPC with implicit integration. It motivates local dynamics derivatives and
derivatives of implicit integration residuals. Its reported benefit over
numerical differentiation does not establish the benefit over this package's
JAX autodiff implementation.

The earlier PR measurements suggest larger gains for full Jacobians of large,
dense models than for single products. A full Jacobian writes quadratic output;
a directional recurrence can avoid it, but JAX already computes products without
forming that matrix. Therefore dense-Jacobian speedups do not predict rollout
training speedups. Recheck against the rebased equations and both derivative modes.

## Baseline established by the rebase

The PR now includes the analytical recurrence prerequisite from #188 and targets
main directly. The existing shared execution boundary selects the optional PCS
analytical rule. Normal primal execution retains configured backend dispatch.

The analytical recurrence currently supports fixed-base PCS with standard
inertia, identity actuation, frictionless fixed threadlike routing, and supported
threadlike passive elements. Floating bases, custom inertia, friction, and other
component laws use autodiff through current JAX assembly. This preserves the
physics introduced since the original PR. Force-derivative helpers include
installed passive and actuation components, including velocity dependence.

The current rule skips state recurrences for input/external-force-only tangents.
It still calls the mass solve separately for primal and tangent, and mixed
parameter tangents may repeat dynamics preparation. Factorization reuse and
component-level fallback remain implementation work. The 64-velocity threshold
between fused assembly and the contracted inverse-dynamics recurrence is an
existing CPU heuristic, not a device-independent optimum.

## Implementation sequence

### 1. Establish representative performance and correctness gates

Extend `tools/benchmarks/pcs_custom_jvp_speed_test.py` with explicit single-step
VJPs, scalar rollout losses, and independent state/control/parameter activity.
Keep the reference on the protected current JAX primal so it cannot accidentally
differentiate through the analytical forward-dynamics rule under test.

Cover small and large full-strain models, reduced strains, threadlike components,
and a fallback model. Measure primal, JVP, VJP, `jacfwd`, `jacrev`, existing dense
helpers, mixed state/control/parameter derivatives, and multiple directions at
one state. Distinguish an axis of directions from an axis of different states.
Use synchronized timings after compilation; report compile time and peak memory
separately. Compare CPU and the project's supported GPU configuration.

Acceptance: numerical equivalence with independent AD and selected finite
differences, including dot-product transpose checks, JIT, vmap, higher derivatives,
and a short integrated loss. Record regressions and variance, not only winners.
Use realistic rollout lengths before selecting a new default implementation.

### 2. Share primal preparation and the implicit dynamics solve

Introduce private preparation and linear-action helpers under the PCS
implementation; extract their implementation to a private module if this reduces
the size of `pcs.py` without creating another public contract. Preparation should
produce only requested kinematic data, dynamics values, mass factorization, and
the residuals needed by the selected derivative operation.

For acceleration `a`, write `M(q,p) a = r(q,v,u,p)` and use

```text
M da = dr - dM a
M.T lambda = bar_a
bar_z = (partial_z r - partial_z M a).T lambda
```

Here the partial derivative holds `a` fixed for the local implicit identity.
Keep the dependence of saved primal quantities available to higher derivatives;
blanket `stop_gradient` on the cache would invalidate them. Reuse one mass
factorization for the primal and linear solves. For SPD inertia, apply the
Cholesky factors to all requested right-hand sides; retain existing solve
semantics and numerical regularization.

For the solve alone this changes repeated factorization work toward
`O(n^3 + n^2 k)` for `k` directions. It does not change the cost of assembling
kinematics or derivative right-hand sides. Verify sharing in lowered/compiled
programs and timings rather than assuming XLA merges independent solves.

### 3. Make activity selection and reuse explicit

Dispatch using symbolic zero/absent tangents at tracing time. Never test numerical
tangent values to select a derivative algorithm. Combine active state, control,
external-force, and parameter contributions into one acceleration right-hand
side and solve once per direction or block of directions.

| Active derivatives | Required additional work |
|---|---|
| Controls / external forces | Control-force action and mass solve |
| Configuration | Contracted kinematic, inertia, convective, gravity, and force derivatives |
| Velocity | Velocity-dependent convective and force actions |
| Model parameters | AD of uncovered residual dependencies, sharing primal data where possible |
| Several groups | Shared preparation plus the sum of their linear actions |

The current public state is one packed array: symbolic activity is known per
array leaf, not per slice. A zero-filled configuration part of a supplied state
tangent is not automatically a symbolic zero. A private boundary accepting
separate configuration and velocity arguments may expose more activity when the
caller differentiates those arguments separately. Validate this with tracing
tests before claiming velocity-only pruning.

For several directions at the same state, map the linear action over shared
residuals. For different states, each state needs its own preparation. Add tests
that fail if inactive recurrence helpers run, and inspect factorization counts
for mixed tangents and multiple directions. Keep separate performance tests for
explicit numerical zeros, which must remain correct without guaranteed pruning.

### 4. Compare a HiJax boundary with the Equinox custom JVP

Prototype the preparation/action split on actual PCS forward dynamics using
`jax.experimental.hijax.HiPrim` on JAX 0.12.1. A `lin` rule can return a primal
and shared residuals; `linearized` applies their linear map. Derive JVP/VJP rules
from that map initially. Supply batching that preserves the primitive and reuse
for relevant axes; merely expanding into generic operations under vmap can lose
the intended optimization.

An isolated two-coordinate implicit-solve prototype already passes JIT, mixed
JVPs, selective VJPs, both Jacobian modes, a Hessian, reusable linearization, and
batched directions. Its activity masks include controls-only and state-only
cases. It demonstrates API feasibility, not PCS speed or complete batching
support. The next experiment must compare against both current Equinox rules
and plain AD on the workloads above.

Adopt HiJax only after a measurable benefit or a material simplification is
established. Keep all experimental API use in one private adapter. Its API
changed between installed JAX 0.11.0 and tested 0.12.1, so an adopted version
requires explicit dependency/CI coverage, Python compatibility review, and
matching GPU plugins. An environment upgrade is allowed. Avoid supporting two
permanent analytical implementations once the experiment chooses a route.

### 5. Extend composition across terms and inverse dynamics

Reuse the same local spatial algebra and force actions in forward dynamics,
inverse dynamics, and dynamics terms. The present inverse-dynamics derivative
helpers define `ID = M a + C v + G`; document that convention. If a public inverse
dynamics evaluator returns the generalized effort required after passive forces,
define it explicitly as `M a + C v + G + elastic + damping` and compose those
contributions. Mapping generalized effort back to actuator controls is a separate
possibly underactuated solve.

An acceleration tangent in inverse dynamics is simply `M da`; state and parameter
directions reuse existing actions. For forward dynamics prefer `dM a` and `d(Cv)`
when these contractions are all that is needed. A request for the derivative of
the mass matrix itself still requires its actual output derivative.

Replace the current whole-model fallback incrementally with component-level AD
fallbacks only where the analytical equations remain valid. An actuator extension
should not require copying the entire dynamics rule. Floating-coordinate
kinematics and custom inertia need their own validated analytical support before
enabling those recurrences. Continue to differentiate continuous model leaves;
topology and discretization choices remain static.

### 6. Validate integration and memory behavior

Test normal JAX differentiation through an explicit step and a short scan first,
then the integration/adjoint modes used by the package. A continuous dynamics
Jacobian is not the discrete step Jacobian. An implicit step requires the
derivative of its converged residual and its own linear solve. Adaptive solver
and adjoint choices also determine which derivatives are supported.

Measure saved-residual memory over rollout length. Save expensive reusable data
only when its backward use warrants the memory; checkpoint selected recomputable
kinematic data otherwise. Add a custom transpose only if the derived VJP is
measurably inadequate. Add hardware-specific kernels only after profiling
identifies a persistent bottleneck in the composed workload.

## Recent JAX facilities relevant to this plan

| Facility | Role and limitation |
|---|---|
| HiJax `lin`, activity masks, symbolic zeros | Explicit preparation reuse and selective derivative work; experimental API |
| First-class VJP objects and `jax.fwd_and_bwd` | Inspect and carry backward residuals across compiled boundaries; useful internally |
| Custom rematerialization | Control rollout residual storage/recomputation after memory profiling |
| `lax.polynomial` (0.12.0) | Candidate for scalar polynomial coefficients; not a drop-in replacement for noncommutative matrix-polynomial derivatives |
| XLA inlining/compiler controls | Tune observed compilation/fusion issues; do not replace the algebraic sharing design |
| Pallas | Later targeted kernels if device profiling justifies added maintenance |

Direct linearization has already been the default since JAX 0.7. Established
`custom_linear_solve` can express implicit solve differentiation, but it does
not itself make two independently requested factorizations share storage.

References: [JAX changelog](https://docs.jax.dev/en/latest/changelog.html),
[HiJax derivative rules](https://docs.jax.dev/en/latest/301/custom-derivatives.html),
[first-class VJPs](https://docs.jax.dev/en/latest/301/vjp-objects.html),
[custom linear solves](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html),
[Equinox custom JVPs](https://docs.kidger.site/equinox/api/transformations/#equinox.filter_custom_jvp),
[Diffrax adjoints](https://docs.kidger.site/diffrax/api/adjoints/).

## Rebase validation

CPU validation with JAX 0.11.0 / Equinox 0.13.7 passed the PCS tests
(27 dynamics/derivative tests plus 106 remaining tests), 84 spatial derivative
primitive tests, 19 benchmark-harness tests, five new compatibility tests, and
three selected execution/actuation tests. The extended threadlike regression
also checks the full Jacobian and force helpers against AD.

An isolated JAX 0.12.1 / Equinox 0.13.8 environment passed the same 19
benchmark-harness and five compatibility tests. The HiJax feasibility probe
passed separately. This is CPU evidence; the inherited CUDA plugin was from
0.11.0 and incompatible with the newer jaxlib, so GPU upgrade validation remains
outstanding. The package dependency floor is unchanged by this rebase.

### Fresh CPU timing sample

JAX 0.11.0, FP64, five Gauss points, pinned to CPU core 7. Each result is
synchronized. Speedup is AD time divided by analytical time. This is a focused
sample of the rebased equations, not a repeat of the previous 24-model sweep.

| Setup | Segments | Velocities | JVP speedup | Full Jacobian speedup |
|---|---:|---:|---:|---:|
| Full strain | 1 | 6 | 1.11x | 1.11x |
| Full strain | 16 | 96 | 1.25x | 3.10x |
| Threadlike, reduced strain | 1 | 2 | 1.09x | 1.23x |
| Threadlike, reduced strain | 16 | 32 | 1.04x | 1.16x |

The main sample used three warmups and five samples of five calls. The two
one-segment JVP entries above come from a follow-up with five warmups and 20
samples of 50 calls. Their initial short samples were 0.77x and 0.70x, so small
timings are sensitive to measurement conditions and need repeated steady-state
measurement. The largest scale-normalized difference in the main sample was
`3.18e-11`. These measurements support prioritizing the largest dense Jacobians
and investigating reuse for products; they establish no VJP or GPU speedup.

```bash
taskset -c 7 env PYTHONPATH=src JAX_PLATFORMS=cpu python \
  tools/benchmarks/pcs_custom_jvp_speed_test.py \
  --segment-counts 1 16 --setups full threadlike \
  --workloads state_jvp state_jacobian \
  --warmup-runs 3 --repeats 5 --calls-per-sample 5

taskset -c 7 env PYTHONPATH=src JAX_PLATFORMS=cpu python \
  tools/benchmarks/pcs_custom_jvp_speed_test.py \
  --segment-counts 1 --setups full threadlike --workloads state_jvp \
  --warmup-runs 5 --repeats 20 --calls-per-sample 50
```
