from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union

from rendering.vercors.environment_settings import env
from objects.ast.models import Expression, Primary, VariableRef, Composite, Assignment

if TYPE_CHECKING:
    from objects.ast.interfaces import SlcoStatementNode
    from objects.locking.models import LockingInstruction


def render_locking_instruction(model: LockingInstruction) -> str:
    """
    Render the given lock instruction as Java code.
    """
    if not model.has_locks():
        # Simply return an empty string if no locks need to be rendered.
        return ""

    return ""


# Conversion from SLCO to Java operators.
java_operator_mappings = {
    "<>": "!=",
    "=": "==",
    "and": "&&",
    "or": "||",
    "not": "!"
}


def render_expression(model: Expression, control_node_methods: List[str], control_node_method_prefix: str) -> str:
    """
    Render the given expression object as in-line Java code.
    """
    if model.op == "**":
        left_str = render_expression_component(model.values[0], control_node_methods, control_node_method_prefix)
        right_str = render_expression_component(model.values[1], control_node_methods, control_node_method_prefix)
        return "(int) Math.pow(%s, %s)" % (left_str, right_str)
    elif model.op == "%":
        # The % operator in Java is the remainder operator, which is not the modulo operator.
        left_str = render_expression_component(model.values[0], control_node_methods, control_node_method_prefix)
        right_str = render_expression_component(model.values[1], control_node_methods, control_node_method_prefix)
        return "Math.floorMod(%s, %s)" % (left_str, right_str)
    else:
        values_str = [
            render_expression_component(v, control_node_methods, control_node_method_prefix) for v in model.values
        ]
        return (" %s " % java_operator_mappings.get(model.op, model.op)).join(values_str)


def render_primary(model: Primary, control_node_methods: List[str], control_node_method_prefix: str) -> str:
    """
    Render the given primary object as in-line Java code.
    """
    if model.value is not None:
        exp_str = str(model.value).lower()
    elif model.ref is not None:
        exp_str = render_expression_component(model.ref, control_node_methods, control_node_method_prefix)
    else:
        exp_str = "(%s)" % render_expression_component(model.body, control_node_methods, control_node_method_prefix)
    return ("!(%s)" if model.sign == "not" else model.sign + "%s") % exp_str


def render_variable_ref(model: VariableRef, control_node_methods: List[str], control_node_method_prefix: str) -> str:
    """
    Render the given variable reference object as in-line Java code.
    """
    result = model.var.name
    if model.index is not None:
        result += "[%s]" % render_expression_component(model.index, control_node_methods, control_node_method_prefix)
    return result


def render_expression_component(
    model: SlcoStatementNode, control_node_methods: List[str] = None, control_node_method_prefix: str = ""
) -> str:
    """
    Render the given statement object as in-line Java code and create supportive methods if necessary.
    """
    if control_node_methods is None:
        control_node_methods = []

    return ""


def create_statement_prefix(transition_prefix: str, i: int) -> Tuple[str, int]:
    """
    Generate a unique prefix for the statement.
    """
    return f"{transition_prefix}_s_{i}", i + 1


def render_root_expression(
    model: Union[Expression, Primary], control_node_methods: List[str], transition_prefix: str, i: int
) -> Tuple[str, int]:
    """
    Render the given expression object as Java code.
    """
    return "", i


def render_assignment(
    model: Assignment, control_node_methods: List[str], transition_prefix: str, i: int
) -> Tuple[str, int]:
    """
    Render the given assignment object as Java code.
    """
    return "", i


def render_composite(
    model: Composite, control_node_methods: List[str], transition_prefix: str, i: int
) -> Tuple[str, int]:
    """
    Render the given composite object as Java code.
    """
    return "", i


# Add supportive filters.
env.filters["render_statement"] = render_expression_component
env.filters["render_locking_instruction"] = render_locking_instruction

# Import the appropriate templates.
# vercors_control_node_method_template = env.get_template("statements/java_statement_wrapper_method.jinja2template")
# vercors_locking_instruction_template = env.get_template("locking/java_locking_instruction.jinja2template")
# vercors_locking_check_template = env.get_template("locking/java_locking_check.jinja2template")
# vercors_assignment_template = env.get_template("statements/java_assignment.jinja2template")
# vercors_expression_template = env.get_template("statements/java_expression.jinja2template")
# vercors_composite_template = env.get_template("statements/java_composite.jinja2template")
# vercors_transition_template = env.get_template("java_transition.jinja2template")
