from __future__ import annotations

from typing import TYPE_CHECKING

import settings
from rendering.java.model_renderer import render_type
from rendering.vercors.environment_settings import env
from objects.ast.models import Composite, Assignment, Expression, Primary
from rendering.vercors.statement_renderer import render_vercors_composite, render_vercors_assignment, \
    render_vercors_root_expression

if TYPE_CHECKING:
    from objects.ast.models import StateMachine, Class, SlcoModel, Transition





def render_vercors_transition(model: Transition) -> str:
    """Render the SLCO state machine as Java code."""

    # Data needed for rendering control nodes as methods.
    transition_prefix = f"t_{model.source}_{model.id}"
    control_node_methods = []
    i = 0

    # Render each of the statements in sequence.
    rendered_statements = []
    for s in model.statements:
        # The first statement is the guard statement.
        if isinstance(s, Composite):
            result, i = render_vercors_composite(s, control_node_methods, transition_prefix, i, model.parent)
        elif isinstance(s, Assignment):
            result, i = render_vercors_assignment(s, control_node_methods, transition_prefix, i, model.parent)
        elif isinstance(s, (Expression, Primary)):
            result, i = render_vercors_root_expression(s, control_node_methods, transition_prefix, i, model.parent)
        else:
            raise Exception(f"No function exists to turn objects of type {type(s)} into Java statements.")
        if result is not None:
            rendered_statements.append(result)

    # Render the transition and its statements.
    return vercors_transition_template.render(
        model=model,
        control_node_methods=control_node_methods,
        rendered_statements=rendered_statements,
        c=model.parent.parent,
        sm=model.parent
    )


def render_vercors_state_machine(model: StateMachine) -> str:
    """Render the SLCO state machine as Java code."""
    return vercors_state_machine_template.render(
        model=model
    )


def render_vercors_class(model: Class) -> str:
    """Render the SLCO class as Java code."""
    return vercors_class_template.render(
        model=model
    )


def render_vercors_model(model: SlcoModel) -> str:
    """Render the SLCO model as Java code."""
    return vercors_model_template.render(
        model=model
    )


# Add supportive filters.
env.filters["render_vercors_transition"] = render_vercors_transition
env.filters["render_vercors_state_machine"] = render_vercors_state_machine
env.filters["render_vercors_class"] = render_vercors_class
env.filters["render_type"] = render_type

# Import the appropriate templates.
vercors_transition_template = env.get_template("vercors_transition.jinja2template")
vercors_state_machine_template = env.get_template("vercors_state_machine.jinja2template")
vercors_class_template = env.get_template("vercors_class.jinja2template")
vercors_model_template = env.get_template("vercors_model.jinja2template")
