"""Compute kinematics, dynamics, and energy derivatives with ordinary transformations.

Run this script normally for autodiff. On CPU, --custom-derivatives opts in to
custom rules. The large full-strain state JVP is a recommended CPU workload.
The kinematics and energy examples normally use the default autodiff. Select the mode before
the first compiled call; changing the setting does not change cached traces.
"""

import argparse

import jax
import jax.numpy as jnp

from soromox import custom_jvp_enabled, set_custom_jvp_enabled
from soromox.systems import PCS, LinkSpec, PCSStructure


def main():
    """Run the three derivative examples with the selected mode.

    Returns:
        None. Prints derivative shapes and norms for each example.
    """
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

    # Kinematics: sensitivity of a backbone pose to configuration.
    pose_direction = jax.jit(
        lambda q: jax.jvp(
            lambda q: robot.forward_kinematics(q, jnp.array(0.75)),
            (q,),
            (jnp.full_like(q, 0.01),),
        )[1]
    )(q).block_until_ready()

    result = state_jvp(state, direction).block_until_ready()
    # Energy: gradients with respect to configuration and generalized velocity.
    energy_gradients = jax.jit(jax.grad(robot.total_energy, argnums=(0, 1)))(q, v)
    jax.block_until_ready(energy_gradients)
    mode = "custom derivatives" if custom_jvp_enabled() else "autodiff (default)"
    print(f"Mode: {mode}; device: {jax.devices()[0].platform}")
    print(f"Pose JVP shape: {pose_direction.shape}")
    print(
        f"State JVP shape: {result.shape}; norm: {float(jnp.linalg.norm(result)):.6g}"
    )

    print(f"Energy gradient shapes: {[value.shape for value in energy_gradients]}")


if __name__ == "__main__":
    main()
