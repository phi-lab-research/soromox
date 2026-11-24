"""Optional JAX 0.12.1 HiJax comparison on the actual PCS equations.

This is an experiment, not a production dependency. It asks whether explicitly
carrying a linearization changes the cost of the existing private derivative
composition. The primitive remains present when mapping different states; its
conservative scan batching is intended for correctness, not GPU throughput.
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
from jax.experimental.hijax import HiPrim, instantiate_zeros, jvp_from_lin, vjp_from_lin

from tools.benchmarks import pcs_custom_jvp_speed_test as benchmark


class Dynamics(HiPrim):
    def __init__(self, function, avals):
        self.in_avals = avals
        self.out_aval = avals[0]  # packed state derivative
        self.params = {"function": function}
        super().__init__()

    def expand(self, *args):
        return self.params["function"](*args)

    def lin(self, nzs_in, *primals):
        active = tuple(i for i, needed in enumerate(nzs_in) if needed)

        def evaluate(*variables):
            values = list(primals)
            for index, value in zip(active, variables, strict=True):
                values[index] = value
            return self.expand(*values)

        primal, action = jax.linearize(evaluate, *(primals[i] for i in active))
        # The mask is static metadata in the callable rather than a saved array.
        return primal, (
            jax.tree_util.Partial(lambda fn, *t: fn(*(t[i] for i in active)), action),
        )

    def linearized(self, residuals, *tangents):
        return residuals[0](*(instantiate_zeros(t) for t in tangents))

    jvp = jvp_from_lin
    vjp_fwd, vjp_bwd_retval = vjp_from_lin

    def batch(self, axis_data, args, dims):
        values = tuple(
            jnp.broadcast_to(x, (axis_data.size, *x.shape))
            if dim is None
            else jnp.moveaxis(x, dim, 0)
            for x, dim in zip(args, dims, strict=True)
        )
        return jax.lax.map(lambda values: self(*values), values), 0


def run(setup="threadlike", segments=1):
    model = benchmark.build_model(setup, segments, 5)
    inputs = benchmark.build_inputs(model)
    args = (inputs.y, inputs.u, inputs.tau_ext, inputs.system_parameters)

    def dynamics(y, u, force, parameters):
        updated = benchmark.model_with_system_parameters(model, parameters)
        return updated.forward_dynamics(0.0, y, (u, force))

    primitive = Dynamics(dynamics, tuple(jax.typeof(x) for x in args))
    direction = (
        inputs.y_direction,
        inputs.u_direction,
        jnp.ones_like(inputs.tau_ext),
        inputs.system_parameter_direction,
    )

    def workloads(fn):
        return {
            "state_jvp": lambda y, u, f, p: jax.jvp(
                lambda y: fn(y, u, f, p), (y,), (direction[0],)
            ),
            "mixed_jvp": lambda *args: jax.jvp(fn, args, direction),
            "mixed_vjp": lambda *args: jax.value_and_grad(
                lambda *a: direction[0] @ fn(*a), argnums=(0, 1, 2, 3)
            )(*args),
            "directions": lambda y, u, f, p: jax.vmap(
                jax.linearize(lambda y: fn(y, u, f, p), y)[1]
            )(jnp.stack([jnp.roll(direction[0], i) for i in range(4)])),
        }

    direct, hijax = workloads(dynamics), workloads(primitive)
    rows = []
    for name in direct:
        a = benchmark.measure(direct[name], args, 5, 10, 10)
        b = benchmark.measure(hijax[name], args, 5, 10, 10)
        _, error = benchmark.difference(a.output, b.output)
        if error > 1e-6:
            raise AssertionError((name, error))
        rows.append(
            {
                "workload": name,
                "ordinary_s": a.execution,
                "hijax_s": b.execution,
                "speedup": a.execution / b.execution,
                "error": error,
                "ordinary_buffers": a.buffer_bytes,
                "hijax_buffers": b.buffer_bytes,
            }
        )
    # Check higher derivatives and batching at distinct states without erasing
    # the primitive's derivative rules in its batching implementation.
    for transform in (jax.jacfwd, jax.jacrev, jax.hessian):

        def fn(y):
            return jnp.sum(primitive(y, *args[1:]))

        def ref(y):
            return jnp.sum(dynamics(y, *args[1:]))

        _, error = benchmark.difference(transform(ref)(args[0]), transform(fn)(args[0]))
        if error > 1e-6:
            raise AssertionError((transform, error))
    states = jnp.stack((args[0], args[0] + 1e-3))
    for transform in (lambda f: f, jax.jacrev):
        fn = jax.vmap(transform(lambda y: primitive(y, *args[1:])))
        ref = jax.vmap(transform(lambda y: dynamics(y, *args[1:])))
        _, error = benchmark.difference(ref(states), fn(states))
        if error > 1e-6:
            raise AssertionError(("batch", error))
    print(json.dumps(rows, indent=2))
    print("PASS: derivatives, higher derivatives, and distinct-state batching")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", choices=benchmark.SETUPS, default="threadlike")
    parser.add_argument("--segments", type=int, default=1)
    options = parser.parse_args()
    run(options.setup, options.segments)
