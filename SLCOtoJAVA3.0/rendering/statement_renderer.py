from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Set

import settings
from objects.ast.util import get_class_variable_references
from rendering.environment_settings import env
from objects.ast.models import Expression, Primary, VariableRef, Composite, Assignment

if TYPE_CHECKING:
    from objects.ast.interfaces import SlcoStatementNode
    from objects.locking.models import LockingInstruction


def render_locking_check(class_variable_references: Set[VariableRef]) -> str:
    """
    Render code that checks if all of the locks used by the model have been acquired.
    """
    return java_locking_check_template.render(
        class_variable_references=class_variable_references
    )


def render_locking_instruction(model: LockingInstruction) -> str:
    """
    Render the given lock instruction as Java code.
    """
    if not model.has_locks():
        # Simply return an empty string if no locks need to be rendered.
        return ""

    # Render the appropriate lock acquisitions and releases through the appropriate template.
    return java_locking_instruction_template.render(
        model=model
    )


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
        left_str = render_statement(model.values[0], control_node_methods, control_node_method_prefix)
        right_str = render_statement(model.values[1], control_node_methods, control_node_method_prefix)
        return "(int) Math.pow(%s, %s)" % (left_str, right_str)
    elif model.op == "%":
        # The % operator in Java is the remainder operator, which is not the modulo operator.
        left_str = render_statement(model.values[0], control_node_methods, control_node_method_prefix)
        right_str = render_statement(model.values[1], control_node_methods, control_node_method_prefix)
        return "Math.floorMod(%s, %s)" % (left_str, right_str)
    else:
        values_str = [render_statement(v, control_node_methods, control_node_method_prefix) for v in model.values]
        return (" %s " % java_operator_mappings.get(model.op, model.op)).join(values_str)


def render_primary(model: Primary, control_node_methods: List[str], control_node_method_prefix: str) -> str:
    """
    Render the given primary object as in-line Java code.
    """
    if model.value is not None:
        exp_str = str(model.value).lower()
    elif model.ref is not None:
        exp_str = render_statement(model.ref, control_node_methods, control_node_method_prefix)
    else:
        exp_str = "(%s)" % render_statement(model.body, control_node_methods, control_node_method_prefix)
    return ("!(%s)" if model.sign == "not" else model.sign + "%s") % exp_str


def render_variable_ref(model: VariableRef, control_node_methods: List[str], control_node_method_prefix: str) -> str:
    """
    Render the given variable reference object as in-line Java code.
    """
    result = model.var.name
    if model.index is not None:
        result += "[%s]" % render_statement(model.index, control_node_methods, control_node_method_prefix)
    return result


def render_statement(
        model: SlcoStatementNode,
        control_node_methods: List[str] = None,
        control_node_method_prefix: str = ""
) -> str:
    """
    Render the given statement object as in-line Java code and create supportive methods if necessary.
    """
    if control_node_methods is None:
        control_node_methods = []

    # Construct the in-line statement.
    if isinstance(model, Expression):
        in_line_statement = render_expression(model, control_node_methods, control_node_method_prefix)
    elif isinstance(model, Primary):
        in_line_statement = render_primary(model, control_node_methods, control_node_method_prefix)
    elif isinstance(model, VariableRef):
        in_line_statement = render_variable_ref(model, control_node_methods, control_node_method_prefix)
    else:
        raise Exception(f"No function exists to turn objects of type {type(model)} into in-line Java statements.")

    # Statements with an atomic node that has locks needs to be rendered as a method.
    # Create the method and refer to it instead.
    if model.locking_atomic_node is not None and (
        model.locking_atomic_node.has_locks() or (
            settings.verify_locks and len(model.locking_atomic_node.child_atomic_nodes) == 0
        )
    ):
        # Give the node a name, render it as a separate method, and return a call to the method.
        method_name = f"{control_node_method_prefix}_n_{len(control_node_methods)}"

        # Get the class variables used in the object.
        class_variable_references = get_class_variable_references(model) if settings.verify_locks else set()

        # Render the statement as a control node method and return the method name.
        control_node_methods.append(
            java_control_node_method_template.render(
                locking_control_node=model.locking_atomic_node,
                method_name=method_name,
                in_line_statement=in_line_statement,
                class_variable_references=class_variable_references
            )
        )
        return f"{method_name}()"
    else:
        # Return the statement as an in-line Java statement.
        return in_line_statement


def create_statement_prefix(transition_prefix: str, i: int) -> Tuple[str, int]:
    """
    Generate a unique prefix for the statement.
    """
    return f"{transition_prefix}_s_{i}", i + 1


def render_root_expression(
        model: Expression,
        control_node_methods: List[str],
        transition_prefix: str,
        i: int,
        is_guard: bool,
        exclude_comment: bool = False
) -> Tuple[str, int]:
    """
    Render the given expression object as Java code.
    """
    # Create an unique prefix for the statement.
    statement_prefix, i = create_statement_prefix(transition_prefix, i)

    # Create an in-line Java code string for the expression.
    in_line_expression = render_statement(model, control_node_methods, statement_prefix)

    # Render the expression as an if statement.
    result = java_expression_template.render(
        model=model,
        in_line_expression=in_line_expression,
        exclude_comment=exclude_comment,
        is_guard=is_guard
    )
    return result, i


def render_assignment(
        model: Assignment,
        control_node_methods: List[str],
        transition_prefix: str,
        i: int,
        exclude_comment: bool = False
) -> Tuple[str, int]:
    """
    Render the given assignment object as Java code.
    """
    # Create an unique prefix for the statement.
    statement_prefix, i = create_statement_prefix(transition_prefix, i)

    # Create an in-line Java code string for the left and right hand side.
    in_line_lhs = render_statement(model.left, control_node_methods, statement_prefix)
    in_line_rhs = render_statement(model.right, control_node_methods, statement_prefix)

    class_variable_references = get_class_variable_references(model.left) if settings.verify_locks else set()
    if settings.verify_locks and len(model.locking_atomic_node.child_atomic_nodes) == 0:
        class_variable_references.update(get_class_variable_references(model.right))

    # Render the assignment as Java code.
    result = java_assignment_template.render(
        model=model,
        locking_control_node=model.locking_atomic_node,
        in_line_lhs=in_line_lhs,
        in_line_rhs=in_line_rhs,
        class_variable_references=class_variable_references,
        exclude_comment=exclude_comment
    )
    return result, i


def render_composite(
        model: Composite,
        control_node_methods: List[str],
        transition_prefix: str,
        i: int,
        is_guard: bool,
) -> Tuple[str, int]:
    """
    Render the given composite object as Java code.
    """
    # Gather all the statements used in the composite.
    rendered_statements = []
    rendered_guard, i = render_root_expression(
        model.guard, control_node_methods, transition_prefix, i, is_guard, exclude_comment=True
    )
    rendered_statements.append(rendered_guard)
    for a in model.assignments:
        rendered_assignment, i = render_assignment(a, control_node_methods, transition_prefix, i, exclude_comment=True)
        rendered_statements.append(rendered_assignment)

    # Render the composite and all of its statements.
    result = java_composite_template.render(
        model=model,
        rendered_statements=rendered_statements
    )
    return result, i


# Add supportive filters.
env.filters["render_statement"] = render_statement
env.filters["render_locking_instruction"] = render_locking_instruction
env.filters["render_locking_check"] = render_locking_check

# Import the appropriate templates.
java_control_node_method_template = env.get_template("statements/java_statement_wrapper_method.jinja2template")
java_locking_instruction_template = env.get_template("locking/java_locking_instruction.jinja2template")
java_locking_check_template = env.get_template("locking/java_locking_check.jinja2template")
java_assignment_template = env.get_template("statements/java_assignment.jinja2template")
java_expression_template = env.get_template("statements/java_expression.jinja2template")
java_composite_template = env.get_template("statements/java_composite.jinja2template")
java_transition_template = env.get_template("java_transition.jinja2template")
