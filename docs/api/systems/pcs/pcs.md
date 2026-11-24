# Spatial PCS

The general Piecewise Constant Strain (PCS) implementation provides the core modeling framework for continuum soft robots, based on the discrete Cosserat approach proposed by Renda et al. (2018).

## Overview

This module contains the fundamental PCS implementation that serves as the foundation for more specialized PCS variants. It provides the core mathematical framework for modeling continuum robots using piecewise constant strain assumptions, following the discrete Cosserat approach for multisection soft manipulator dynamics.

## Construction and updates

The shared component model is described in
[Continuum Robot Components](../continuum-components.md). For full immutable
replacement and optimization workflows, see
[Parameters, Updates, and Optimization](../../../user-guide/parameters-and-optimization.md).

```python
import jax.numpy as jnp

from soromox.systems import LinkSpec, PCS

pcs = PCS.from_links([
    LinkSpec.circular(
        length=0.2,
        radius=0.012,
        density=1000.0,
        young_modulus=1.0e6,
        shear_modulus=3.4e5,
        material_damping_coefficient=1.0e4,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
])

# Explicit generalized matrices bypass isotropic material construction.
explicit = PCS.from_links([
    LinkSpec.circular(
        length=0.15,
        radius=0.01,
        density=1000.0,
        stiffness=jnp.diag(jnp.array([0.2, 0.8, 0.8, 100.0, 30.0, 30.0])),
        damping=jnp.diag(jnp.array([0.01, 0.02, 0.02, 0.5, 0.1, 0.1])),
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
])

pcs = pcs.update_link_params(
    stiffness=1.1 * pcs.params.link.stiffness,
    damping=0.9 * pcs.params.link.damping,
)
replacement = pcs.params.link.replace(density=1.05 * pcs.params.link.density)
pcs = pcs.with_params(pcs.params.replace(link=replacement))
```

## Dynamics derivatives

Use ordinary JAX transformations to differentiate forward dynamics, dynamics
terms, or inverse dynamics. The derivative rules compose with a loss or an
integration routine and preserve differentiation of continuous model parameters.

```python
import jax

# Directional state derivative and reverse-mode control derivative.
yd, dyd = jax.jvp(lambda y: pcs.forward_dynamics(t, y, (u,)), (y,), (dy,))
bar_u = jax.grad(lambda u: cotangent @ pcs.forward_dynamics(t, y, (u,)))(u)

# Required generalized effort and its configuration/velocity/acceleration Jacobians.
effort = pcs.inverse_dynamics(q, qd, qdd)
Dq, Dv, Da = jax.jacfwd(pcs.inverse_dynamics, argnums=(0, 1, 2))(q, qd, qdd)
```

`inverse_dynamics` returns `M @ qdd + Cqd + G + elastic + damping @ qd`, including
installed passive elements. External applied forces must be subtracted from
that effort. Converting the remaining generalized effort to actuator controls
is a separate problem, particularly for an underactuated robot.

The older `inverse_dynamics_derivatives` and `inverse_dynamics_backward_pass`
helpers describe only `M @ qdd + Cqd + G`. Differentiate `inverse_dynamics`
directly when passive-force derivatives are needed.

Supported PCS state derivatives use analytical recurrences. Other dependencies
use composable autodiff fallbacks. The forward-dynamics derivative reuses its
inertia factorization across active contributions. A Jacobian of continuous
forward dynamics is distinct from the Jacobian of an integration step; apply
JAX to the step or rollout when that is the derivative your application needs.

## API Reference

::: soromox.systems.pcs.pcs
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

## References

The PCS (Piecewise Constant Strain) model was originally proposed in:

Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete cosserat approach for multisection soft manipulator dynamics. *IEEE Transactions on Robotics*, 34(6), 1518-1533.
