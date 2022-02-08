from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, Optional

import settings
from rendering.java.model_renderer import render_type
from rendering.vercors.environment_settings import env
from objects.ast.models import Composite, Assignment, Expression, Primary
from rendering.vercors.statement_renderer import render_vercors_composite, render_vercors_assignment, \
    render_vercors_root_expression

if TYPE_CHECKING:
    from objects.ast.models import StateMachine, Class, SlcoModel, Transition, Variable


def render_vercors_transition_verification(
        model: Transition, verification_targets: Dict[Variable, List[int]]
) -> Tuple[str, List[str]]:
    """
    Render the VerCors statements that verify whether the transition adheres to the set behavior (guard, assignment).
    """
    # Create a list of support variables.
    support_variables: List[str] = ["boolean _guard"]
    for v, targets in verification_targets.items():
        type_name = "boolean" if v.is_boolean else "int"
        for i in targets:
            support_variables.append(f"{type_name} _rhs_{i}")
            if v.is_array:
                support_variables.append(f"{type_name} _index_{i}")

    # Create rules for the expected outcome of the transition.
    return_value_verification = [
        "_guard ==> (\\result == true)",
        "!_guard ==> (\\result == false)"
    ]

    # Create rules for the expected changes in variable types, including the support methods if necessary.
    value_change_verification = []
    support_pure_functions = []
    for v, targets in verification_targets.items():
        if v.is_array:
            # Create a function that selects the appropriate value based on the given index.
            # Create the function.
            type_name = "boolean" if v.is_boolean else "int"
            function_name = f"value_{model.source}_{model.id}_{v.name}"
            function_typed_parameters = ", ".join(f"int _index_{i}, {type_name} _rhs_{i}" for i in targets)

            # Build the function from the ground up, such that most recent assignments have priorities.
            function_body = f"(_i == _index_{targets[0]}) ? _rhs_{targets[0]} : v_old"
            for i in targets[1:]:
                function_body = f"(_i == _index_{i}) ? _rhs_{i} : ({function_body})"
            function_declaration = f"pure {type_name} {function_name}(int _i, {function_typed_parameters}, " \
                                   f"{type_name} v_old) = {function_body}"
            support_pure_functions.append(function_declaration)

            # Use the variable name with a possible prefix due to it being a class variable.
            v_name = f"c.{v.name}" if v.is_class_variable else v.name

            # Create the function call.
            function_parameters = ", ".join(f"_index_{i}, _rhs_{i}" for i in targets)
            function_call = f"{function_name}(_i, {function_parameters}, \\old({v_name}[_i]))"

            # Add the appropriate constructs to verify the value change.
            value_change_verification.append(
                f"_guard ==> (\\forall* int _i; 0 <= _i && _i < {v_name}.length; {v_name}[_i] == {function_call})"
            )
            value_change_verification.append(
                f"!_guard ==> (\\forall* int _i; 0 <= _i && _i < {v_name}.length; {v_name}[_i] == \\old({v_name}[_i]))"
            )
        else:
            # Only consider the last target.
            value_change_verification.append(f"_guard ==> ({v.name} == _rhs_{targets[-1]})")
            value_change_verification.append(f"!_guard ==> ({v.name} == \\old({v.name}))")

    # Create the verification contract.
    return vercors_transition_verification_statements_template.render(
        support_variables=support_variables,
        return_value_verification=return_value_verification,
        value_change_verification=value_change_verification
    ), support_pure_functions


def render_vercors_transition(model: Transition) -> str:
    """Render the SLCO state machine as Java code."""

    # Data needed for rendering control nodes as methods.
    transition_prefix = f"t_{model.source}_{model.id}"
    control_node_methods = []
    i = 0

    # Keep a dictionary of assignment operations that need to be verified by the transition's contract.
    verification_targets: Dict[Variable, List[int]] = dict()

    # Render each of the statements in sequence.
    rendered_statements = []
    for s in model.statements:
        # The first statement is the guard statement.
        if isinstance(s, Composite):
            result, i = render_vercors_composite(
                s, control_node_methods, transition_prefix, i, model.parent, verification_targets
            )
        elif isinstance(s, Assignment):
            result, i = render_vercors_assignment(
                s, control_node_methods, transition_prefix, i, model.parent, verification_targets
            )
        elif isinstance(s, (Expression, Primary)):
            result, i = render_vercors_root_expression(
                s, control_node_methods, transition_prefix, i, model.parent
            )
        else:
            raise Exception(f"No function exists to turn objects of type {type(s)} into Java statements.")
        if result is not None:
            rendered_statements.append(result)

    # Render the transition verification statements.
    transition_verification, support_pure_functions = render_vercors_transition_verification(
        model,
        verification_targets
    )

    # Render the transition and its statements.
    return vercors_transition_template.render(
        model=model,
        control_node_methods=control_node_methods,
        rendered_statements=rendered_statements,
        c=model.parent.parent,
        sm=model.parent,
        transition_verification=transition_verification,
        support_pure_functions=support_pure_functions
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

vercors_transition_verification_statements_template = env.get_template(
    "util/vercors_transition_verification_statements.jinja2template"
)
