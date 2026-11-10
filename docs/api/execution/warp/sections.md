# Warp Rod Section API

`execution.warp.sections` evaluates the same geometry and constitutive laws as
[`RodConstitutiveLaw`](../../systems/continuum-components.md#rod-constitutive-laws)
using native FP64 Warp functions. It supports circular solids and annuli,
rectangular solids and uniform-wall tubes, and solid ellipses, with the same
supported combinations of uniform/effective shear and polar/Saint-Venant torsion.

## Placement in a solver

Sectional mechanics belongs to parameter preparation: evaluate it when geometry
or material changes, project it through the strain basis, then reuse the resulting
inertia, stiffness and damping during stepping. The existing GVS dynamics kernels
already consume those prepared quantities. Normal `GVS` construction and immutable
parameter updates continue to use differentiable JAX preparation, regardless of
the selected dynamics backend. No additional JAX/Warp boundary is introduced there.

For a native Warp integration, use `rod_section_function` to compose the section
calculation into an existing preparation kernel, or `rod_section_kernel` to process
a batch. The factories cache only geometry, constitutive choices and whether a
cavity exists. Dimensions and the elastic Poisson ratio remain runtime inputs.
Group sections by these static choices and launch each group over views into
caller-owned buffers. Sampling a tapered profile uses the same kernels as a
constant section.

Both factories reject unsupported geometry/law combinations before compilation.
As with the other native APIs, the caller validates numerical inputs before
launching: dimensions must be positive and finite, cavities must lie inside the
outer boundary, rectangular walls must have uniform thickness, and Cowper shear
requires `-1 < nu <= 0.5`. Solid functions ignore the inner dimensions.

## Section evaluation

```python
import warp as wp
from soromox.execution.warp.sections import rod_section_kernel
from soromox.systems.components.cross_sections import CrossSectionGeometry
from soromox.systems import (
    RodConstitutiveLaw, TransverseShearRodModel, TorsionRodModel,
)

law = RodConstitutiveLaw(
    transverse_shear=TransverseShearRodModel.EFFECTIVE_AREA,
    torsion=TorsionRodModel.SAINT_VENANT,
)
kernel = rod_section_kernel(CrossSectionGeometry.CIRCULAR, law, hollow=True)
device = "cpu"  # or a CUDA device
# vec2d stores radius in slot 0; slot 1 is ignored for circles.
outer = wp.array([[0.03, 0.0], [0.02, 0.0]], dtype=wp.vec2d, device=device)
inner = wp.array([[0.028, 0.0], [0.018, 0.0]], dtype=wp.vec2d, device=device)
nu = wp.array([0.3, 0.3], dtype=wp.float64, device=device)
inertia, young, shear, damping = [
    wp.empty((2, 6), dtype=wp.float64, device=device) for _ in range(4)
]
wp.launch(
    kernel, dim=2, inputs=[outer, inner, nu],
    outputs=[inertia, young, shear, damping], device=device,
)
```

For rectangles, the two dimensions are height along material y and width along z.
For ellipses they are the semi-axes along y and z. All six-vector outputs use
angular-linear ordering. Their physical interpretation is:

| Output | Multiply by | Result |
| --- | --- | --- |
| `inertia` | density | diagonal sectional mass inertia |
| `young` | Young's modulus E | axial and bending stiffness contribution |
| `shear` | shear modulus G | torsional and transverse shear stiffness contribution |
| `damping` | material damping coefficient eta | sectional Kelvin–Voigt damping |

Stiffness is `E * young + G * shear`. Cowper shear uses the elastic Poisson ratio
in both the stiffness and damping outputs. Rotational mass inertia always uses
the polar area moment, independently of the torsion law.

## GVS projection and differentiation

`project_rod_material_kernel` integrates the three material diagonals through a
sampled strain basis. Pack each link's samples contiguously, with an INT32 offsets
array of length `num_links + 1`. Supply basis shape `(num_samples, 6, max_dof)`
and physical quadrature weights (including link length). Include the model's
rotational length scaling in the basis and zero-pad inactive coordinates.
Launch over `(num_links, max_dof, max_dof)` to write the three unit operators.
Links may use different quadrature counts and sectional laws. Form each link's
generalized matrices as `K = E * K_E + G * K_G` and `D = eta * D_eta`.
Explicit user-provided matrices retain their own ownership; apply this projection
only where material-derived matrices are wanted.

These kernels support `wp.Tape` gradients through dimensions, Poisson ratio,
sectional diagonals, weights and basis values. Enable `requires_grad` on the
participating arrays and retain intermediate buffers. If Poisson ratio is derived
from E and G, compute `nu = E / (2 * G) - 1` within the differentiated pipeline.
Changing E or G then requires refreshing the Cowper operators as well as scaling
them. The numerical test suite compares these adjoints with JAX.

Launches allocate no buffers and perform no host transfers. Warm the selected
kernels and allocate all buffers before CUDA graph capture; retain the arrays
for the graph's lifetime. Numerical validation is the caller's responsibility
outside capture.

::: soromox.execution.warp.sections
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
