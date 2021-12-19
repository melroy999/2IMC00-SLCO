from __future__ import annotations

from typing import TYPE_CHECKING, Union

import settings
from rendering.environment_settings import env
from objects.ast.models import Expression, Composite, Assignment
from rendering.statement_renderer import render_composite, render_assignment, render_root_expression

if TYPE_CHECKING:
    from objects.ast.models import Transition, StateMachine, Class, SlcoModel, Variable, DecisionNode, Object, GuardNode


def render_type(model: Variable):
    """Render the type of the given variable object."""
    if model.is_array:
        return "%s[]" % ("boolean" if model.is_boolean else "int")
    else:
        return "%s" % ("boolean" if model.is_boolean else "int")


def is_decision_node(model):
    """Return true when the given model is a decision node, false otherwise."""
    return isinstance(model, DecisionNode)


def is_transition(model):
    """Return true when the given model is a transition, false otherwise."""
    return isinstance(model, Transition)


def render_transition(model: Transition) -> str:
    """
    Render the given transition as a Java method, including methods for the control nodes when appropriate.
    """
    # Data needed for rendering control nodes as methods.
    transition_prefix = f"t_s_{model.source}_{model.id}"
    control_node_methods = []
    i = 0

    # Render each of the statements in sequence.
    rendered_statements = []
    for s in model.statements:
        if isinstance(s, Composite):
            result, i = render_composite(s, control_node_methods, transition_prefix, i)
        elif isinstance(s, Assignment):
            result, i = render_assignment(s, control_node_methods, transition_prefix, i)
        elif isinstance(s, Expression):
            result, i = render_root_expression(s, control_node_methods, transition_prefix, i)
        else:
            raise Exception(f"No function exists to turn objects of type {type(s)} into Java statements.")
        rendered_statements.append(result)

    # Render the transition and its statements.
    return java_transition_template.render(
        model=model,
        control_node_methods=control_node_methods,
        rendered_statements=rendered_statements
    )


def render_decision_structure_node(model: Union[DecisionNode, GuardNode]):
    """Render a decision node fitting the desired parameters."""
    if isinstance(model, GuardNode):
        return java_guard_template.render(model=model)
    else:
        if model.is_deterministic:
            return java_deterministic_decision_template.render(model=model)
        else:
            return java_non_deterministic_decision_template.render(model=model)


def render_object_instantiation(model: Object):
    """Render the instantiation of the given object."""
    arguments = []
    for i, a in enumerate(model.initial_values):
        v = model.type.variables[i]
        if v.is_array:
            arguments.append("new %s[]{%s}" % ("boolean" if v.is_boolean else "int", ", ".join(map(str, a))))
        else:
            arguments.append(a)

    return java_object_instantiation.render(
        name=model.type.name,
        arguments=arguments
    )


def render_state_machine(model: StateMachine):
    """Render the SLCO state machine as Java code."""
    return java_state_machine_template.render(
        model=model,
        settings=settings
    )


def render_class(model: Class):
    """Render the SLCO class as Java code."""
    return java_class_template.render(model=model)


def render_lock_manager(_):
    """Render the lock manager of the model."""
    return java_lock_manager_template.render(
        settings=settings
    )


def render_model(model: SlcoModel):
    """Render the SLCO model as Java code."""
    return java_model_template.render(model=model)


# Add supportive filters.
env.filters["render_transition"] = render_transition
env.filters["render_decision_structure_node"] = render_decision_structure_node
env.filters["render_state_machine"] = render_state_machine
env.filters["render_class"] = render_class
env.filters["render_lock_manager"] = render_lock_manager
env.filters["render_type"] = render_type
env.filters["render_object_instantiation"] = render_object_instantiation

env.filters["is_decision_node"] = is_decision_node
env.filters["is_transition"] = is_transition

# Import the appropriate templates.
java_transition_template = env.get_template("java_transition.jinja2template")
java_state_machine_template = env.get_template("java_state_machine.jinja2template")
java_class_template = env.get_template("java_class.jinja2template")
java_model_template = env.get_template("java_model.jinja2template")

java_guard_template = env.get_template("objects/control_flow_node/java_guard.jinja2template")
java_deterministic_decision_template = env.get_template(
    "objects/control_flow_node/java_deterministic_decision.jinja2template"
)
java_non_deterministic_decision_template = env.get_template(
    "objects/control_flow_node/java_non_deterministic_decision.jinja2template"
)

java_lock_manager_template = env.get_template("locking/java_lock_manager.jinja2template")

java_object_instantiation = env.get_template("util/java_object_instantiation.jinja2template")
