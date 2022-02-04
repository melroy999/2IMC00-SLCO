from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from objects.ast.models import Expression, Assignment, Composite
    from objects.ast.interfaces import SlcoStatementNode


def get_expression_wrapper_comment(model: SlcoStatementNode):
    """
    Create an easily identifiable comment for expression wrapper methods.
    """
    # Create an easily identifiable comment.
    statement_comment = f"// SLCO expression wrapper | {model}"
    return statement_comment


def get_root_expression_comment(is_superfluous, model: Expression):
    """
    Create an easily identifiable comment for root expression statements.
    """
    # Create an easily identifiable comment.
    original_slco_expression_string = str(model.get_original_statement())
    preprocessed_slco_expression_string = str(model)
    statement_comment = "// "
    if is_superfluous:
        statement_comment += "(Superfluous) "
    statement_comment += f"SLCO expression | {original_slco_expression_string}"
    if original_slco_expression_string != preprocessed_slco_expression_string:
        statement_comment += f" -> {preprocessed_slco_expression_string}"
    return statement_comment


def get_assignment_comment(model: Assignment):
    """
    Create an easily identifiable comment for assignment statements.
    """
    # Create an easily identifiable comment.
    original_slco_statement_string = str(model.get_original_statement())
    preprocessed_slco_statement_string = str(model)
    statement_comment = f"// SLCO assignment | {original_slco_statement_string}"
    if original_slco_statement_string != preprocessed_slco_statement_string:
        statement_comment += f" -> {preprocessed_slco_statement_string}"
    return statement_comment


def get_composite_comment(model: Composite):
    """
    Create an easily identifiable comment for composite statements.
    """
    # Create an easily identifiable comment.
    original_slco_composite_string = str(model.get_original_statement())
    preprocessed_slco_composite_string = str(model)
    statement_comment = f"// SLCO composite | {original_slco_composite_string}"
    if original_slco_composite_string != preprocessed_slco_composite_string:
        statement_comment += f" -> {preprocessed_slco_composite_string}"
    return statement_comment
