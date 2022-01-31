from __future__ import annotations

from typing import TYPE_CHECKING

import settings
from rendering.java.model_renderer import render_type
from rendering.vercors.environment_settings import env

if TYPE_CHECKING:
    from objects.ast.models import StateMachine, Class, SlcoModel, Transition


def render_transition(model: Transition) -> str:
    """Render the SLCO state machine as Java code."""
    return vercors_transition_template.render(
        model=model,
        control_node_methods=[]
    )


def render_state_machine(model: StateMachine) -> str:
    """Render the SLCO state machine as Java code."""
    return vercors_state_machine_template.render(
        model=model
    )


def render_class(model: Class) -> str:
    """Render the SLCO class as Java code."""
    return vercors_class_template.render(
        model=model
    )


def render_lock_manager(model: Class) -> str:
    """Render the lock manager of the model."""
    return vercors_lock_manager_template.render(
        model=model
    )


def render_model(model: SlcoModel) -> str:
    """Render the SLCO model as Java code."""
    return vercors_model_template.render(
        model=model
    )


# Add supportive filters.
env.filters["render_transition"] = render_transition
env.filters["render_state_machine"] = render_state_machine
env.filters["render_class"] = render_class
env.filters["render_lock_manager"] = render_lock_manager
env.filters["render_type"] = render_type

# Import the appropriate templates.
vercors_transition_template = env.get_template("vercors_transition.jinja2template")
vercors_state_machine_template = env.get_template("vercors_state_machine.jinja2template")
vercors_class_template = env.get_template("vercors_class.jinja2template")
vercors_model_template = env.get_template("vercors_model.jinja2template")

vercors_lock_manager_template = env.get_template("locking/vercors_lock_manager.jinja2template")
