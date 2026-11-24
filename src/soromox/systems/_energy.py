"""Energy derivative composition using the model's force and inertia hooks."""

import jax.numpy as jnp

from soromox.systems._dynamics import has_tangent


def evaluate(model, q, velocity, kind):
    """Evaluate protected energy equations with the model's state layout."""
    if model.floating_base:
        if kind == "elastic":
            return model._elastic_energy(model.split_configuration(q)[1])
        if kind in ("potential", "total"):
            value = model._gravitational_energy(q) + model._elastic_energy(
                model.split_configuration(q)[1]
            )
            return (
                value + model._kinetic_energy(q, velocity) if kind == "total" else value
            )
    primal = getattr(model, f"_{kind}_energy")
    return primal(q, velocity) if kind in ("kinetic", "total") else primal(q)


def energy_jvp(primals, tangents):
    """Compose state actions; defer joint parameter directions to one AD pass.

    Floating states also use the shared AD fallback. State-only requests use
    the model's force and inertia identities and skip inactive contributions.
    """
    model, q, velocity, kind = primals
    dmodel, dq, dv, _ = tangents
    if model.floating_base or has_tangent(dmodel):
        return None
    value = evaluate(*primals)
    direction = jnp.zeros_like(value)
    if kind in ("kinetic", "total"):
        if dq is not None:
            direction = direction + model._kinetic_energy_q_jvp_tangent(q, velocity, dq)
        if dv is not None:
            direction = direction + model._kinetic_energy_qd_jvp_tangent(
                q, velocity, dv
            )
    if dq is not None and kind != "kinetic":
        if kind == "gravitational":
            force = model.gravitational_force(q)
        elif kind == "elastic":
            force = model.elastic_force(q)
        else:
            force = model._potential_energy_gradient(q)
        direction = direction + force @ dq
    return value, direction
