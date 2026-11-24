"""Compute a PCS state JVP with default autodiff or explicit custom derivatives.

Run this script normally for autodiff. On CPU, --custom-derivatives opts in to
custom rules for the large full-strain model used here. Select the mode before
the first compiled call; changing the setting does not change cached traces.
"""

import argparse

import jax
import jax.numpy as jnp

from soromox import custom_jvp_enabled, set_custom_jvp_enabled
from soromox.systems import PCS, LinkSpec, PCSStructure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-derivatives", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if args.custom_derivatives:
        set_custom_jvp_enabled(True)

    robot = PCS.from_links(
        [
            LinkSpec.circular(
                length=0.1,
                radius=0.02,
                density=1050.0,
                young_modulus=6e5,
                shear_modulus=2.5e5,
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            for _ in range(16)
        ],
        gravity=jnp.array([0.0, 0.0, -9.81]),
        structure=PCSStructure(num_gauss_points=3),
        backend="jax",
    )
    q = jnp.linspace(-0.02, 0.03, robot.num_coordinates)
    v = jnp.zeros(robot.num_velocities)
    state = robot.pack_state(q, v)
    controls = jnp.zeros(robot.num_actuators)
    direction = jnp.full_like(state, 0.01)

    @jax.jit
    def state_jvp(state, direction):
        return jax.jvp(
            lambda state: robot.forward_dynamics(0.0, state, (controls,)),
            (state,),
            (direction,),
        )[1]

    result = state_jvp(state, direction).block_until_ready()
    mode = "custom derivatives" if custom_jvp_enabled() else "autodiff (default)"
    print(f"Mode: {mode}; device: {jax.devices()[0].platform}")
    print(
        f"State JVP shape: {result.shape}; norm: {float(jnp.linalg.norm(result)):.6g}"
    )


if __name__ == "__main__":
    main()
