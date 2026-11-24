"""Capabilities shared by analytical actuator state derivatives."""

from soromox.actuation.core import (
    Actuator,
    DirectEffort,
    IdentityActuator,
    IdentityTransmission,
)
from soromox.actuation.threadlike import ThreadlikeActuator, ThreadlikeRouting


def supports_analytical_actuator(actuator: Actuator) -> bool:
    """Check support for direct-effort state actions with fixed routing.

    Args:
        actuator: Installed actuator whose effort and transmission laws are checked.

    Returns:
        Whether the actuator uses standard direct effort with an identity map or
        frictionless fixed threadlike routing. Model-specific geometric support
        must be checked by the caller.
    """
    return type(actuator.effort_model) is DirectEffort and (
        (
            type(actuator) is IdentityActuator
            and type(actuator.transmission) is IdentityTransmission
        )
        or (
            type(actuator) is ThreadlikeActuator
            and type(actuator.transmission.routing) is ThreadlikeRouting
            and actuator.transmission.friction.is_frictionless
        )
    )
