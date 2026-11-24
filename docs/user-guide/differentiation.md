# Differentiation

Use JAX transformations on SoRoMoX system methods to obtain derivatives of
kinematics, dynamics, energies, and simulated trajectories. The same interface
supports state, control, and continuous model-parameter derivatives.

## Choosing a derivative representation

| Task | JAX operation | Result |
|---|---|---|
| Optimize a scalar objective | `jax.grad` or `jax.value_and_grad` | Gradient with respect to the selected inputs |
| Propagate an input perturbation | `jax.jvp` | Jacobian-vector product |
| Propagate an output sensitivity | `jax.vjp` | Vector-Jacobian product |
| Supply a linearization matrix to a controller or solver | `jax.jacfwd` or `jax.jacrev` | Full Jacobian |

Use products for directional sensitivities and gradients of scalar losses.
Form a full matrix when the downstream algorithm needs its entries, such as
an MPC solver that consumes discrete-time state and input matrices.

The examples below assume an existing `robot`, a valid configuration `q`,
velocity `v`, packed state `y = robot.pack_state(q, v)`, controls `u`, and time `t`.
Choose tangent directions `dq`, `dv`, and `dy` with the shapes of their respective
inputs, and a scalar backbone coordinate `s`.

```python
import jax
import jax.numpy as jnp

field = lambda state, control: robot.forward_dynamics(t, state, (control,))

# State sensitivity in direction dy.
ydot, dydot = jax.jvp(lambda state: field(state, u), (y,), (dy,))

# Control gradient of a scalar objective.
control_gradient = jax.grad(lambda control: jnp.sum(field(y, control)**2))(u)

# Continuous-time state and control Jacobians.
A, B = jax.jacfwd(field, argnums=(0, 1))(y, u)
```

`jacfwd` assembles columns; `jacrev` assembles rows. Both work with the custom
rules described below. The performance of constructing a full matrix can differ
substantially between the two transformations.

## Choosing a derivative mode

**Use autodiff by default. Enable custom derivatives for the workloads listed
below.** Choose by the differentiated quantity: a dynamics recommendation does
not determine the best mode for a pose, geometric Jacobian, or scalar energy.

### Kinematics gradients

| Query or workload | CPU | GPU |
|---|---|---|
| Pose derivatives with respect to configuration | Autodiff | Autodiff |
| PCS pose derivative with respect to arc length | Autodiff | Custom derivatives |
| PlanarPCS or GVS pose derivative with respect to arc length | Autodiff | Autodiff |
| PCS geometric-Jacobian VJP | Custom derivatives | Custom derivatives |
| PlanarPCS geometric-Jacobian VJP | Custom derivatives | Autodiff |
| Geometric-Jacobian JVP or full derivative matrix | Autodiff | Autodiff |
| Mixed configuration/arc-length or model-parameter derivatives | Autodiff | Autodiff |
| Other kinematics queries | Autodiff | Autodiff |

A pose derivative and a derivative of the geometric Jacobian are different
queries. The latter differentiates `J(q, s)` itself, as needed when a loss or a
controller depends on `J`. A custom JVP can also accelerate its transposed VJP;
small forward-mode gains do not imply small reverse-mode gains.

PCS and planar PCS retain their simple analytical pose arc-length rules in
custom mode. GVS uses AD of its pose recurrence. The equivalent scalar methods
`jacobian()` and `jacobian_inertialframe()` share derivative selection, including
the Jacobian output of scalar pose/Jacobian queries. Specialized spatial batches
retain their own traversal and AD rules.

### Dynamics gradients

| Query or workload | CPU | GPU |
|---|---|---|
| Large, full-strain PCS forward dynamics: full state Jacobian | Custom derivatives | Autodiff |
| Large, full-strain PCS forward dynamics: state JVP or VJP | Custom derivatives | Autodiff |
| Small or reduced-strain models | Autodiff | Autodiff |
| Control-only gradients | Autodiff | Autodiff |
| Mixed state/control/parameter gradients | Autodiff | Autodiff |
| Gradients of rollout losses | Autodiff | Autodiff |
| Other models and derivative queries | Autodiff | Autodiff |

These recommendations use the current PCS forward-dynamics measurements. The
large full-strain case has 96 velocities; this is an example, not a prescribed
size threshold. Measurement conditions and complete results are in the
[derivative comparison](https://github.com/phi-lab-research/soromox/pull/261).

### Energy gradients

| Query or workload | CPU | GPU |
|---|---|---|
| Configuration/velocity gradients of kinetic, gravitational, elastic, potential, or total energy | Autodiff | Autodiff |
| Directional energy derivatives | Autodiff | Autodiff |
| Energy gradients involving model parameters or floating bases | Autodiff | Autodiff |
| Other energy queries | Autodiff | Autodiff |

Energy rules compose by contribution. PCS uses AD for gravitational energy,
including its contribution to potential and total energy, and retains the
kinetic and elastic rules. Kinetic energy and its velocity derivative share the
`inertia_matrix()` computation; these queries do not launch Warp dynamics kernels.
The shared inertia optimization also accelerates ordinary autodiff. With the
remaining gains close or mixed, use autodiff for energy gradients. The custom
rules remain available, and their coverage is listed below.

The kinematics and energy recommendations use PCS with 16 segments, planar PCS
with four segments, and GVS with four segments, in double precision on CPU and
an RTX 5090. They describe isolated queries. Use autodiff for complete rollout
losses and unlisted workloads.

### Selecting the mode

**Autodiff is the package default on CPU and GPU.** Ordinary JAX calls need no
mode configuration:

```python
# Uses autodiff with the default package settings.
gradient = jax.grad(lambda state: jnp.sum(field(state, u)**2))(y)
```

For a workload recommended above, explicitly opt into custom rules before tracing:

```python
from soromox import custom_jvp_mode

with custom_jvp_mode(True):
    state_jvp = jax.jit(
        lambda state, direction: jax.jvp(
            lambda state: field(state, u), (state,), (direction,),
        )[1]
    )
    dydot = state_jvp(y, dy)  # First trace occurs inside the context.
```

The context restores the previous mode when it exits, including on exceptions.
For an application-wide choice, call `set_custom_jvp_enabled(True)` before
tracing. `set_custom_jvp_enabled(False)` restores autodiff. Custom mode combines
selected analytical rules with AD fallbacks; it does not force every derivative
to use an analytical formula.

The runnable example covers pose sensitivities, forward-dynamics state JVPs,
and total-energy gradients:

```bash
python examples/differentiation/derivative_modes.py
```

Use `--custom-derivatives` to exercise the optional rules. For the large PCS
state-JVP workload, select CPU with `JAX_PLATFORMS=cpu`. The recommendations
above apply separately to each query.

The setting is read during tracing. Set it before the first call to a compiled
function; changing it does not change an existing executable. For independent
mode comparisons, use separate processes so cached traces cannot retain the
previous setting. The context manager changes the same process-wide setting.

The switch controls optional analytical rules selected by `jax.grad`,
`jax.jvp`, `jax.vjp`, `jax.jacfwd`, and `jax.jacrev` on system methods. Explicit
analytical helpers, such as `forward_dynamics_state_jacobian`, keep their
specified algorithms. Numerical safety primitives and implicit routing solvers
retain the derivative definitions needed by those operations.

Device, execution backend, and derivative mode are separate choices.
`backend="jax"` selects an execution backend; it does not disable analytical
derivatives. See [Execution Devices and Backends](execution-devices-and-backends.md#differentiation)
for differentiation through methods with Warp primal implementations.

## Analytical derivative support

The tables describe optional rules selected by JAX transformations with custom
mode enabled, for fixed-base models. **Analytical** means a dedicated derivative
formula is available. **Hybrid** combines analytical actions with autodiff.
**AD** means the query uses autodiff. An AD entry still supports differentiation.
Availability and the performance recommendation above are separate.

### Kinematics

Here `J(q, s)` is the geometric velocity Jacobian returned by `jacobian()`.
For PCS, planar PCS, and GVS, `jacobian_inertialframe()` is the equivalent
scalar query. Differentiating `J` computes its derivatives.

| Model or system | Pose with respect to `q` | Pose with respect to `s` | `J` with respect to `q` | `J` with respect to `s` |
|---|---|---|---|---|
| PCS, ISupport | AD | Analytical | Analytical | Analytical |
| PlanarPCS, PlanarHSA | AD | Analytical | Analytical | Analytical |
| GVS, Helicoidal | AD | AD | Analytical | Analytical |
| Pendulum | AD | AD | Analytical | AD |
| ArticulatedSoftRobot, McKibbenActuatedUMArm | AD | AD | Analytical | AD |

Mixed `q`/`s` directions and requests involving model parameters use joint AD.
These entries describe scalar backbone queries. Specialized batched methods can
use AD instead. Floating-base derivatives use AD. Explicit analytical
arc-length helpers remain available for GVS.

### Dynamics

| Model or system | Forward dynamics: state | Inverse dynamics: state | Dynamics terms `(M, Cqd, G)`: state | Inverse dynamics: acceleration |
|---|---|---|---|---|
| PCS | Hybrid | Hybrid | Analytical | Analytical (`M @ da`) |
| ISupport | AD | AD | AD | AD |
| PlanarPCS, PlanarHSA | AD | AD | AD | AD |
| GVS, Helicoidal | AD | AD | AD | AD |
| Pendulum | AD | AD | AD | AD |
| ArticulatedSoftRobot, McKibbenActuatedUMArm | AD | AD | AD | AD |

PCS dynamics rules support the standard fixed-base inertia model. Floating-base
states and custom inertia use AD. Supported state recurrences can be combined
with AD for custom actuator and passive-element laws. Velocity-only dynamics-term
directions use AD. For large state requests, CPU selects a contracted analytical
recurrence and GPU selects AD term actions inside the shared solve rule.

PCS forward dynamics combines control and external-force directions through the
inertia solve. Model-parameter directions use AD of the terms and forces, followed
by the implicit acceleration derivative. Other systems use AD for those queries.
There is no analytical-only mode that forces every dependency onto a closed form.

### Energy

| Model or system | Gravitational energy: `q` | Elastic energy: `q` | Kinetic energy: `(q, v)` | Potential / total energy |
|---|---|---|---|---|
| PCS, ISupport | AD | Analytical | Hybrid | Hybrid |
| PlanarPCS, PlanarHSA | Analytical | Analytical | Hybrid | Analytical / Hybrid |
| GVS, Helicoidal | Analytical | Analytical | Hybrid | Analytical / Hybrid |
| Pendulum | Analytical | Analytical | Hybrid | Analytical / Hybrid |
| ArticulatedSoftRobot, McKibbenActuatedUMArm | Analytical | Analytical | Hybrid | Analytical / Hybrid |

Kinetic-energy velocity derivatives use inertia actions; configuration
derivatives also use an AD convective-force action. Potential and total energy
compose the underlying rules, so PCS gravity can use AD while its kinetic and
elastic contributions retain their custom rules.

Requests involving model parameters use one joint AD evaluation. Floating-base
energy derivatives also use AD. State-only rules omit inactive contributions.
VJPs and full Jacobians are obtained from the same JVP rules through JAX
transformations.

## Kinematics examples

```python
# Pose derivative along the backbone.
pose, pose_s = jax.jvp(
    lambda s_value: robot.forward_kinematics(q, s_value),
    (s,), (jnp.ones_like(s),),
)

# Directional change of the geometric Jacobian.
J, dJ = jax.jvp(lambda position: robot.jacobian(position, s), (q,), (dq,))

```

## Dynamics and inverse dynamics

The dynamics terms are `M`, the convective force `Cqd`, and gravity `G`.
Differentiate `robot.dynamics_terms(q, v)` to obtain their derivatives.

```python
terms, dterms = jax.jvp(robot.dynamics_terms, (q, v), (dq, dv))

# Required generalized effort and its configuration/velocity/acceleration Jacobians.
effort = robot.inverse_dynamics(q, v, acceleration)
Dq, Dv, Da = jax.jacfwd(robot.inverse_dynamics, argnums=(0, 1, 2))(
    q, v, acceleration,
)
```

`inverse_dynamics` returns `M @ acceleration + Cqd + G + passive_force(q, v)`.
It includes elastic and damping effort. Subtract external applied effort before
solving for actuator controls. For inertial effort alone, differentiate
`robot.inverse_dynamics(q, v, acceleration) - robot.passive_force(q, v)`.

## Energy examples

```python
# Gradients of scalar energies.
gravity_gradient = jax.grad(robot.gravitational_energy)(q)
elastic_gradient = jax.grad(robot.elastic_energy)(q)
dE_dq, dE_dv = jax.grad(robot.total_energy, argnums=(0, 1))(q, v)
```

## Differentiating parameter updates

Construct a candidate model through immutable update methods inside the loss:

```python
def density_loss(density):
    candidate = robot.update_link_params(density=density)
    acceleration = candidate.split_state(candidate.forward_dynamics(t, y, (u,)))[1]
    return jnp.mean(acceleration**2)

value, density_gradient = jax.value_and_grad(density_loss)(robot.params.link.density)
```

This example applies to continuum models with link parameters. Numeric physical
parameters are differentiable; topology and other static structural choices are
not. Use the model's update methods so derived quantities stay consistent.
For material PyTrees, see the
[parameter update and optimization examples](parameters-and-optimization.md#differentiating-parameter-updates).

## Differentiating through integration

Differentiate the integration step when the application needs the derivative of
the next state. Differentiating `forward_dynamics` gives the derivative of the
continuous-time vector field.

```python
def euler_step(state, control):
    return state + dt * robot.forward_dynamics(t, state, (control,))

A_step, B_step = jax.jacfwd(euler_step, argnums=(0, 1))(y, u)

def rollout_loss(control):
    def advance(state, _):
        next_state = euler_step(state, control)
        return next_state, None

    final_state, _ = jax.lax.scan(advance, y, xs=None, length=steps)
    return jnp.mean((final_state - target_state)**2)

loss, gradient = jax.value_and_grad(rollout_loss)(u)
```

The illustrative Euler update assumes fixed-base Euclidean coordinates. For
floating-base poses, use an integrator that preserves the configuration
representation. Solver adjoints and checkpointing determine how a longer rollout
stores or recomputes intermediate states; the selected system derivative rules
compose with that integration strategy.
