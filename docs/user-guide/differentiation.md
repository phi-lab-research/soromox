# Differentiation

Use JAX transformations on system methods for derivatives of kinematics,
dynamics, energies, and simulated trajectories. State, control, and continuous
model parameters are differentiable.

## Choosing a representation

| Task | JAX operation | Result |
|---|---|---|
| Optimize a scalar loss | `grad` or `value_and_grad` | Input gradient |
| Propagate an input perturbation | `jvp` | Jacobian-vector product |
| Propagate an output sensitivity | `vjp` | Vector-Jacobian product |
| Supply matrices to a controller or solver | `jacfwd` or `jacrev` | Full Jacobian |

Use products for directional sensitivities and scalar losses. Form full matrices
when the algorithm consumes their entries, as in many MPC linearizations.

## Choosing a derivative mode

**Use autodiff by default.** Enable custom derivatives for the combinations below.

### Kinematics

| Query | Model | CPU | GPU |
|---|---|---|---|
| Pose derivative with respect to arc length | PCS | Autodiff | Custom |
| Geometric-Jacobian VJP | PlanarPCS | Custom | Autodiff |
| Geometric-Jacobian VJP | PCS | Custom | Custom |
| All other kinematics derivatives | All models | Autodiff | Autodiff |

A geometric-Jacobian derivative differentiates the matrix `J(q, s)` itself.

### Dynamics

| Query | Model | CPU | GPU |
|---|---|---|---|
| Forward-dynamics full state Jacobian | Large, full-strain PCS | Custom | Autodiff |
| Forward-dynamics state JVP or VJP | Large, full-strain PCS | Custom | Autodiff |
| Control, mixed-input, parameter, or rollout gradients | All models | Autodiff | Autodiff |
| All other dynamics derivatives | All models | Autodiff | Autodiff |

### Energy

Use **autodiff for all energy gradients**, including configuration, velocity,
parameter, and directional derivatives.

Recommendations use double-precision CPU and RTX 5090 measurements, including
16-segment full-strain PCS, four-segment PlanarPCS, and four-segment GVS models.
See [measurement details](https://github.com/phi-lab-research/soromox/pull/261).

## Selecting the mode

Autodiff is the package default on CPU and GPU. For a recommended custom workload,
select the mode before the first compiled call:

```python
import jax

from soromox import custom_jvp_mode

with custom_jvp_mode(True):
    state_jvp = jax.jit(lambda y, dy: jax.jvp(
        lambda state: robot.forward_dynamics(t, state, (u,)), (y,), (dy,),
    )[1])
    dydot = state_jvp(y, dy)
```

The context restores the previous mode on exit. For an application-wide choice,
use `set_custom_jvp_enabled(True)`; `False` restores autodiff. Custom mode combines
analytical actions with AD fallbacks. The setting is process-wide and captured
when tracing; existing compiled functions retain their mode.

The switch applies to ordinary JAX transformations of system methods. Explicit
analytical helpers such as `forward_dynamics_state_jacobian`, numerical-safety
primitives, and implicit solvers retain their specified derivative algorithms.
Execution backend selection is independent; see
[Execution Devices and Backends](execution-devices-and-backends.md#differentiation).

## Analytical support

These tables describe custom mode for fixed-base models. **Analytical** means a
dedicated derivative formula; **Hybrid** combines analytical actions and AD;
**AD** uses autodiff. All entries support differentiation.

### Kinematics

`J` is the geometric velocity Jacobian. For scalar PCS, PlanarPCS, and GVS
queries, `jacobian()` and `jacobian_inertialframe()` share derivative selection.

| Model | Pose: `q` | Pose: `s` | `J`: `q` | `J`: `s` |
|---|---|---|---|---|
| Pendulum | AD | AD | Analytical | AD |
| ArticulatedSoftRobot, McKibbenActuatedUMArm | AD | AD | Analytical | AD |
| PlanarPCS, PlanarHSA | AD | Analytical | Analytical | Analytical |
| PCS, ISupport | AD | Analytical | Analytical | Analytical |
| GVS, Helicoidal | AD | AD | Analytical | Analytical |

Mixed `q`/`s`, model-parameter, and floating-base requests use AD. Specialized
spatial batches can also use AD.

### Dynamics

| Model | Forward dynamics: state | Inverse dynamics: state | Terms `(M, Cqd, G)`: state | Inverse dynamics: acceleration |
|---|---|---|---|---|
| Pendulum | AD | AD | AD | AD |
| ArticulatedSoftRobot, McKibbenActuatedUMArm | AD | AD | AD | AD |
| PlanarPCS, PlanarHSA | AD | AD | AD | AD |
| PCS | Hybrid | Hybrid | Analytical | Analytical |
| ISupport | AD | AD | AD | AD |
| GVS, Helicoidal | AD | AD | AD | AD |

PCS rules require standard fixed-base inertia. Custom actuator/passive laws use
local AD; floating bases and custom inertia use general AD. Velocity-only term
directions use AD. Large state products use contracted recurrences on CPU and
AD term actions within the shared solve on GPU. Control/external-force directions
share the inertia solve; parameter directions differentiate terms and forces.

### Energy

| Model | Gravitational: `q` | Elastic: `q` | Kinetic: `(q, v)` | Potential / total |
|---|---|---|---|---|
| Pendulum | Analytical | Analytical | Hybrid | Analytical / Hybrid |
| ArticulatedSoftRobot, McKibbenActuatedUMArm | Analytical | Analytical | Hybrid | Analytical / Hybrid |
| PlanarPCS, PlanarHSA | Analytical | Analytical | Hybrid | Analytical / Hybrid |
| PCS, ISupport | AD | Analytical | Hybrid | Hybrid |
| GVS, Helicoidal | Analytical | Analytical | Hybrid | Analytical / Hybrid |

Kinetic rules use inertia and convective-force actions. Potential/total energies
compose their contributions. Parameter and floating-base requests use joint AD.

## Examples

The snippets assume valid `q`, velocity `v`, state `y = robot.pack_state(q, v)`,
controls `u`, acceleration `a`, time `t`, backbone coordinate `s`, and matching
tangent directions.

```python
import jax
import jax.numpy as jnp

# Kinematics: a pose sensitivity along the backbone.
pose, pose_s = jax.jvp(lambda s: robot.forward_kinematics(q, s),
                       (s,), (jnp.ones_like(s),))

# Dynamics: continuous-time state and control matrices.
field = lambda y, u: robot.forward_dynamics(t, y, (u,))
A, B = jax.jacfwd(field, argnums=(0, 1))(y, u)
terms, dterms = jax.jvp(robot.dynamics_terms, (q, v), (dq, dv))
Dq, Dv, Da = jax.jacfwd(robot.inverse_dynamics, argnums=(0, 1, 2))(q, v, a)

# Energy: gradients of a scalar objective.
dE_dq, dE_dv = jax.grad(robot.total_energy, argnums=(0, 1))(q, v)
```

`inverse_dynamics(q, v, a)` returns `M @ a + Cqd + G + passive_force(q, v)`.
Subtract external effort before allocating actuator controls.

For discrete-time matrices or rollout losses, differentiate the integration step
or complete rollout. For example, a fixed-base Euler step is
`step = lambda y, u: y + dt * field(y, u)`. Floating-base poses need an integrator
that preserves their representation. For parameter gradients, construct candidates
through immutable [parameter updates](parameters-and-optimization.md#differentiating-parameter-updates)
inside the differentiated function.

Run `python examples/differentiation/derivative_modes.py` for all three families
of queries; add `--custom-derivatives` to enable the optional rules.
