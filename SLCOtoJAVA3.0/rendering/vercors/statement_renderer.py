from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union, Dict, Optional

import settings
from objects.ast.util import get_variables_to_be_locked
from rendering.common.statement_renderer import get_expression_wrapper_comment, get_root_expression_comment, \
    get_assignment_comment, get_composite_comment
from rendering.vercors.environment_settings import env
from objects.ast.models import Expression, Primary, VariableRef, Composite, Assignment, StateMachine, Class, Variable

if TYPE_CHECKING:
    from objects.ast.interfaces import SlcoStatementNode


# Conversion from SLCO to Java operators.
java_operator_mappings = {
    "<>": "!=",
    "=": "==",
    "and": "&&",
    "or": "||",
    "not": "!"
}


def render_vercors_permissions(c: Class, sm: StateMachine) -> str:
    """Render method contract statements that ensure full permissions over the used variables."""
    return vercors_permissions_template.render(
        c=c,
        sm=sm,
        has_array_c_variables=any(v.is_array for v in c.variables),
        has_array_sm_variables=any(v.is_array for v in sm.variables)
    )


def render_vercors_expression(
        model: Expression,
        sm: StateMachine,
        control_node_methods: List[str],
        control_node_method_prefix: str,
        assumptions: List[str],
        create_control_nodes: bool
) -> str:
    """
    Render the given expression object as in-line Java code.
    """
    if model.op == "**":
        left_str = render_vercors_expression_component(
            model.values[0], sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )
        right_str = render_vercors_expression_component(
            model.values[1], sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )
        return "(int) Math.pow(%s, %s)" % (left_str, right_str)
    elif model.op == "%":
        # The % operator in Java is the remainder operator, which is not the modulo operator.
        left_str = render_vercors_expression_component(
            model.values[0], sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )
        right_str = render_vercors_expression_component(
            model.values[1], sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )
        return "Math.floorMod(%s, %s)" % (left_str, right_str)
    else:
        values_str = [
            render_vercors_expression_component(
                v, sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
            ) for v in model.values
        ]
        return (" %s " % java_operator_mappings.get(model.op, model.op)).join(values_str)


def render_vercors_primary(
        model: Primary,
        sm: StateMachine,
        control_node_methods: List[str],
        control_node_method_prefix: str,
        assumptions: List[str],
        create_control_nodes: bool
) -> str:
    """
    Render the given primary object as in-line Java code.
    """
    if model.value is not None:
        exp_str = str(model.value).lower()
    elif model.ref is not None:
        exp_str = render_vercors_expression_component(
            model.ref, sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )
    else:
        exp_str = "(%s)" % render_vercors_expression_component(
            model.body, sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )
    return ("!(%s)" if model.sign == "not" else model.sign + "%s") % exp_str


def render_vercors_variable_ref(
        model: VariableRef,
        sm: StateMachine,
        control_node_methods: List[str],
        control_node_method_prefix: str,
        assumptions: List[str],
        create_control_nodes: bool
) -> str:
    """
    Render the given variable reference object as in-line Java code.
    """
    result = model.var.name
    if model.var.is_class_variable:
        result = f"c.{result}"
    if model.index is not None:
        result += "[%s]" % render_vercors_expression_component(
            model.index, sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )
    return result


def render_vercors_expression_component(
        model: SlcoStatementNode,
        sm: StateMachine,
        control_node_methods: List[str] = None,
        control_node_method_prefix: str = "",
        assumptions: List[str] = None,
        create_control_nodes: bool = True
) -> str:
    """
    Render the given statement object as in-line Java code and create supportive methods if necessary.
    """
    # Render all atomic node holding expressions as a function--this is required due to a need to verify conjunctions
    # and disjunctions in a specific way because of limitations in VerCors.
    if create_control_nodes and model.locking_atomic_node is not None:
        # Construct the conditional block(s) that need to be included in the statement.
        conditional_blocks = []

        # Render the full statement without control nodes for verification purposes.
        full_simple_in_line_statement = render_vercors_expression_component(model, sm, create_control_nodes=False)

        # Render conjunctions and disjunctions differently due to a lack of short circuit evaluation support in VerCors.
        if isinstance(model, Expression) and model.op in ["and", "or"]:
            # Create statement-local assumptions.
            local_assumptions: List[str] = list(assumptions) if assumptions is not None else []

            for i, v in enumerate(model.values):
                # Find the in-line conditional.
                in_line_statement = render_vercors_expression_component(
                    v, sm, control_node_methods, control_node_method_prefix, local_assumptions, create_control_nodes
                )

                # Invert the in-line conditional if the expression is a conjunction.
                if model.op == "and":
                    in_line_statement = f"!({in_line_statement})"

                # Render the target statement without control nodes for verification purposes.
                simple_in_line_statement = render_vercors_expression_component(v, sm, create_control_nodes=False)

                # Define the appropriate pre- and post conditions.
                post_condition_body = ("!(%s)" if model.op == "and" else "%s") % full_simple_in_line_statement
                post_condition = f"!({simple_in_line_statement})" if model.op == "or" else simple_in_line_statement

                # Render the target branch of the conditional.
                conditional_blocks.append(vercors_expression_if_statement_template.render(
                    in_line_statement=in_line_statement,
                    post_condition_body=post_condition_body,
                    post_condition=post_condition,
                    invert_return_values=(model.op == "and")
                ))

                # Add the outcome as an assumption to succeeding clauses in the conjunction/disjunction.
                local_assumptions.append(post_condition)

            # Only add a post condition if the statement is a conjunction or disjunction.
            post_condition = ("%s" if model.op == "and" else "!(%s)") % full_simple_in_line_statement
        else:
            in_line_statement = get_in_line_statement(
                model, sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
            )

            # Define the appropriate pre- and post conditions.
            post_condition_body = full_simple_in_line_statement
            post_condition = f"!({full_simple_in_line_statement})"

            # Render the target branch of the conditional.
            conditional_blocks.append(vercors_expression_if_statement_template.render(
                in_line_statement=in_line_statement,
                post_condition_body=post_condition_body,
                post_condition=post_condition,
                invert_return_values=False
            ))

            # Only add a post condition if the statement is a conjunction or disjunction.
            post_condition = None

        # Give the node a name, render it as a separate method, and return a call to the method.
        method_name = f"{control_node_method_prefix}_n_{len(control_node_methods)}"

        # Render the statement as a control node method and return the method name.
        control_node_methods.append(
            vercors_control_node_method_template.render(
                assumptions=assumptions,
                target_result=full_simple_in_line_statement,
                statement_comment=get_expression_wrapper_comment(model),
                method_name=method_name,
                conditional_blocks=conditional_blocks,
                invert_return_values=(isinstance(model, Expression) and model.op == "and"),
                post_condition=post_condition,
                c=sm.parent,
                sm=sm,
                # required_lock_requests=
                # model.locking_atomic_node.entry_node.locking_instructions.requires_lock_requests,
                # ensured_lock_requests_success=
                # model.locking_atomic_node.success_exit.locking_instructions.ensures_lock_requests,
                # ensured_lock_requests_failure=
                # model.locking_atomic_node.failure_exit.locking_instructions.ensures_lock_requests,
            )
        )
        return f"{method_name}()"
    else:
        # Get the in-line statement and return it outright.
        in_line_statement = get_in_line_statement(
            model, sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )

        # Return the statement as an in-line Java statement.
        return in_line_statement


def get_in_line_statement(
        model: SlcoStatementNode,
        sm: StateMachine,
        control_node_methods: List[str],
        control_node_method_prefix: str,
        assumptions: List[str],
        create_control_nodes: bool
):
    """
    Generate the correct in-line statement connected to the given object's type.
    """
    # Find the in-line conditional.
    if isinstance(model, Expression):
        in_line_statement = render_vercors_expression(
            model, sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )
    elif isinstance(model, Primary):
        in_line_statement = render_vercors_primary(
            model, sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )
    elif isinstance(model, VariableRef):
        in_line_statement = render_vercors_variable_ref(
            model, sm, control_node_methods, control_node_method_prefix, assumptions, create_control_nodes
        )
    else:
        raise Exception(f"No function exists to turn objects of type {type(model)} into in-line Java statements.")
    return in_line_statement


def create_statement_prefix(transition_prefix: str, i: int) -> Tuple[str, int]:
    """
    Generate a unique prefix for the statement.
    """
    return f"{transition_prefix}_s_{i}", i + 1


def render_vercors_root_expression(
        model: Union[Expression, Primary],
        control_node_methods: List[str],
        transition_prefix: str,
        i: int,
        sm: StateMachine
) -> Tuple[str, int]:
    """
    Render the given expression object as Java code.
    """
    # False expressions should have been filtered out during the decision structure construction.
    if model.is_false():
        # TODO: Check separately whether the statement holds false.
        raise Exception("An illegal attempt is made at rendering an expression that is always false.")

    # Support flags to control rendering settings.
    is_superfluous = False

    # Determine if the statement needs to be rendered or not.
    if model.is_true():
        if isinstance(model, Expression):
            raise Exception("An illegal attempt is made at rendering an expression instead of a true-valued primary.")
        elif model.value is not True:
            raise Exception("An illegal attempt is made at rendering a true primary that is not true-valued.")
        elif not model.locking_atomic_node.has_locks():
            # The if-statement is superfluous.
            is_superfluous = True

    # Create an in-line Java code string for the expression.
    in_line_expression = ""
    if not is_superfluous:
        # Create an unique prefix for the statement.
        statement_prefix, i = create_statement_prefix(transition_prefix, i)
        in_line_expression = render_vercors_expression_component(model, sm, control_node_methods, statement_prefix)

    # Verify that the original statement still holds.
    original_statement = model.get_original_statement()
    in_line_original = render_vercors_expression_component(original_statement, sm, create_control_nodes=False)

    # Render the expression as an if statement.
    result = vercors_expression_template.render(
        in_line_expression=in_line_expression,
        in_line_original=in_line_original,
        statement_comment=get_root_expression_comment(is_superfluous, model),
        is_superfluous=is_superfluous
    )
    return result, i


def render_vercors_assignment(
        model: Assignment,
        control_node_methods: List[str],
        transition_prefix: str,
        i: int,
        sm: StateMachine,
        verification_targets: Dict[Variable, List[int]],
        assignment_id: int = 0
) -> Tuple[str, int]:
    """
    Render the given assignment object as Java code.
    """
    # Create an unique prefix for the statement.
    statement_prefix, i = create_statement_prefix(transition_prefix, i)

    # Create an in-line Java code string for the left and right hand side.
    in_line_lhs = render_vercors_expression_component(model.left, sm, control_node_methods, statement_prefix)
    in_line_rhs = render_vercors_expression_component(model.right, sm, control_node_methods, statement_prefix)
    in_line_index = in_line_lhs[in_line_lhs.find("[")+1:in_line_lhs.rfind("]")] if model.left.var.is_array else None

    # Add a verification target.
    variable_writes = verification_targets.get(model.left.var, [])
    variable_writes.append(assignment_id)
    verification_targets[model.left.var] = variable_writes

    # Verify that the original statement still holds.
    original_statement = model.get_original_statement()
    in_line_lhs_original = render_vercors_expression_component(original_statement.left, sm, create_control_nodes=False)
    in_line_rhs_original = render_vercors_expression_component(original_statement.right, sm, create_control_nodes=False)

    # Render the assignment as Java code.
    result = vercors_assignment_template.render(
        in_line_lhs=in_line_lhs,
        in_line_rhs=in_line_rhs,
        in_line_index=in_line_index,
        in_line_lhs_original=in_line_lhs_original,
        in_line_rhs_original=in_line_rhs_original,
        statement_comment=get_assignment_comment(model),
        is_byte_typed=model.left.var.is_byte,
        assignment_number=assignment_id
    )
    return result, i


def render_vercors_composite(
        model: Composite,
        control_node_methods: List[str],
        transition_prefix: str,
        i: int,
        sm: StateMachine,
        verification_targets: Dict[Variable, List[int]]
) -> Tuple[str, int]:
    """
    Render the given composite object as Java code.
    """
    # Gather all the statements used in the composite.
    rendered_statements = []
    rendered_guard, i = render_vercors_root_expression(model.guard, control_node_methods, transition_prefix, i, sm)
    rendered_statements.append(rendered_guard)
    for n, a in enumerate(model.assignments):
        rendered_assignment, i = render_vercors_assignment(
            a, control_node_methods, transition_prefix, i, sm, verification_targets, n
        )
        rendered_statements.append(rendered_assignment)

    # Render the composite and all of its statements.
    result = vercors_composite_template.render(
        model=model,
        rendered_statements=rendered_statements,
        statement_comment=get_composite_comment(model)
    )
    return result, i


# Add supportive filters.
env.filters["render_statement"] = render_vercors_expression_component
env.filters["render_vercors_permissions"] = render_vercors_permissions

# Import the appropriate templates.
vercors_control_node_method_template = env.get_template("statements/vercors_statement_wrapper_method.jinja2template")
vercors_assignment_template = env.get_template("statements/vercors_assignment.jinja2template")
vercors_expression_template = env.get_template("statements/vercors_expression.jinja2template")
vercors_expression_if_statement_template = env.get_template("statements/vercors_expression_if_statement.jinja2template")
vercors_composite_template = env.get_template("statements/vercors_composite.jinja2template")

vercors_permissions_template = env.get_template("util/vercors_permissions.jinja2template")
