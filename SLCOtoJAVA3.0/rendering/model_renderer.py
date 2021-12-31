from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import settings
from rendering.environment_settings import env
from objects.ast.models import Expression, Composite, Assignment, Transition, GuardNode, DecisionNode, Primary
from rendering.statement_renderer import render_composite, render_assignment, render_root_expression, \
    create_statement_prefix, render_statement

if TYPE_CHECKING:
    from objects.ast.models import StateMachine, Class, SlcoModel, Variable, Object, State


def render_type(model: Variable):
    """Render the type of the given variable object."""
    if model.is_array:
        return "%s[]" % ("boolean" if model.is_boolean else "int")
    else:
        return "%s" % ("boolean" if model.is_boolean else "int")


def render_transition(model: Transition) -> str:
    """
    Render the given transition as a Java method, including methods for the control nodes when appropriate.
    """
    # Data needed for rendering control nodes as methods.
    transition_prefix = f"t_{model.source}_{model.id}"
    control_node_methods = []
    i = 0

    # Render each of the statements in sequence.
    rendered_statements = []
    for s in model.statements:
        # The first statement is the guard statement.
        is_guard: bool = len(rendered_statements) == 0
        if isinstance(s, Composite):
            result, i = render_composite(s, control_node_methods, transition_prefix, i, is_guard)
        elif isinstance(s, Assignment):
            result, i = render_assignment(s, control_node_methods, transition_prefix, i)
        elif isinstance(s, (Expression, Primary)):
            result, i = render_root_expression(s, control_node_methods, transition_prefix, i, is_guard)
        else:
            raise Exception(f"No function exists to turn objects of type {type(s)} into Java statements.")
        rendered_statements.append(result)

    # Render the transition and its statements.
    return java_transition_template.render(
        model=model,
        control_node_methods=control_node_methods,
        rendered_statements=rendered_statements
    )


def render_transition_wrapper(model: Transition) -> str:
    """
    Render a wrapper statement for the given transition in Java code.
    """
    return java_transition_wrapper_template.render(
        model=model
    )


def render_guard_node(
        model: GuardNode,
        control_node_methods: List[str],
        decision_structure_prefix: str,
        i: int
) -> Tuple[str, int]:
    """
    Render a guard node within the decision structure as Java code.
    """
    # Create an unique prefix for the statement.
    statement_prefix, i = create_statement_prefix(decision_structure_prefix, i)

    # Create an in-line Java code string for the expression.
    in_line_conditional = render_statement(model.conditional, control_node_methods, statement_prefix)

    # Render the body as Java code.
    if isinstance(model.body, Transition):
        rendered_body = render_transition_wrapper(model.body)
    elif isinstance(model.body, GuardNode):
        rendered_body, i = render_guard_node(model.body, control_node_methods, decision_structure_prefix, i)
    elif isinstance(model.body, DecisionNode):
        rendered_body, i = render_decision_node(model.body, control_node_methods, decision_structure_prefix, i)
    else:
        raise Exception(f"No function exists to turn objects of type {type(model.body)} into in-line Java statements.")

    # Render the guard node as Java code.
    result = java_guard_node_template.render(
        in_line_conditional=in_line_conditional,
        rendered_body=rendered_body,
        conditional=model.conditional
    )
    return result, i


def render_decision_node(
        model: DecisionNode,
        control_node_methods: List[str],
        decision_structure_prefix: str,
        i: int
) -> Tuple[str, int]:
    """
    Render the given decision node as Java code.
    """
    # Render all of the decisions in Java code.
    rendered_decisions = []
    for decision in model.decisions:
        if isinstance(decision, Transition):
            result = render_transition_wrapper(decision)
        elif isinstance(decision, GuardNode):
            result, i = render_guard_node(decision, control_node_methods, decision_structure_prefix, i)
        elif isinstance(decision, DecisionNode):
            result, i = render_decision_node(decision, control_node_methods, decision_structure_prefix, i)
        else:
            raise Exception(
                f"No function exists to turn objects of type {type(decision)} into in-line Java statements."
            )
        rendered_decisions.append(result)

    # Render the decision of the appropriate type as Java code.
    if model.is_deterministic:
        result = java_deterministic_decision_node_template.render(
            rendered_decisions=rendered_decisions
        )
    else:
        result = java_non_deterministic_decision_node_template.render(
            rendered_decisions=rendered_decisions
        )
    return result, i


def render_decision_structure(model: StateMachine, state: State) -> str:
    """
    Render the decision structure for the given state machine and starting state as Java code.
    """
    # Data needed for rendering control nodes as methods.
    decision_structure_prefix = f"ds_{state}"
    control_node_methods = []
    i = 0

    # Find the target decision structure and render it as Java code.
    if state not in model.state_to_decision_node:
        # place a comment instead of no decision structure has been created for the node.
        method_body = f"// There are no transitions starting in state {state}."
    else:
        root_target_node: DecisionNode = model.state_to_decision_node[state]
        method_body, i = render_decision_node(root_target_node, control_node_methods, decision_structure_prefix, i)

    # Render the state decision structure and its statements.
    return java_decision_structure_template.render(
        state=state,
        control_node_methods=control_node_methods,
        method_body=method_body
    )


def render_state_machine(model: StateMachine) -> str:
    """Render the SLCO state machine as Java code."""
    return java_state_machine_template.render(
        model=model,
        settings=settings
    )


def render_class(model: Class) -> str:
    """Render the SLCO class as Java code."""
    max_lock_id = max((v.lock_id + v.type.size for v in model.variables), default=0)

    return java_class_template.render(
        model=model,
        max_lock_id=max_lock_id
    )


def render_object_instantiation(model: Object) -> str:
    """Render the instantiation of the given object."""
    arguments = []
    for i, a in enumerate(model.initial_values):
        v = model.type.variables[i]
        if v.is_array:
            arguments.append("new %s[]{ %s }" % ("boolean" if v.is_boolean else "int", ", ".join(map(str, a)).lower()))
        else:
            arguments.append(a)

    return java_object_instantiation.render(
        name=model.type.name,
        arguments=arguments
    )


def render_lock_manager(_) -> str:
    """Render the lock manager of the model."""
    return java_lock_manager_template.render(
        settings=settings
    )


def render_model(model: SlcoModel) -> str:
    """Render the SLCO model as Java code."""
    return java_model_template.render(model=model)


# Add supportive filters.
env.filters["render_transition"] = render_transition
env.filters["render_state_machine"] = render_state_machine
env.filters["render_class"] = render_class
env.filters["render_lock_manager"] = render_lock_manager
env.filters["render_type"] = render_type
env.filters["render_object_instantiation"] = render_object_instantiation
env.filters["render_decision_structure"] = render_decision_structure

# Import the appropriate templates.
java_transition_template = env.get_template("java_transition.jinja2template")
java_state_machine_template = env.get_template("java_state_machine.jinja2template")
java_class_template = env.get_template("java_class.jinja2template")
java_model_template = env.get_template("java_model.jinja2template")

java_decision_structure_template = env.get_template("decision_structures/java_decision_structure.jinja2template")
java_transition_wrapper_template = env.get_template("decision_structures/java_transition_wrapper.jinja2template")
java_guard_node_template = env.get_template("decision_structures/java_guard_node.jinja2template")
java_deterministic_decision_node_template = env.get_template(
    "decision_structures/java_deterministic_decision_node.jinja2template"
)
java_sequential_decision_node_template = env.get_template(
    "decision_structures/java_sequential_decision_node.jinja2template"
)
java_non_deterministic_decision_node_template = env.get_template(
    "decision_structures/java_non_deterministic_decision_node.jinja2template"
)

java_lock_manager_template = env.get_template("locking/java_lock_manager.jinja2template")

java_object_instantiation = env.get_template("util/java_object_instantiation.jinja2template")
