from objects.ast.models import Expression, Primary, VariableRef, Composite, Assignment, Transition
from objects.ast.util import copy_node


def restructure_expression(e: Expression):
    e.values = [restructure(v) for v in e.values]
    if len(e.values) == 1:
        # Skip superfluous expressions that only contain one element.
        return e.values[0]
    return e


def restructure_primary(e: Primary):
    if e.body is not None:
        e.body = restructure(e.body)
    elif e.ref is not None:
        e.ref = restructure(e.ref)
    return e


def restructure_variable_ref(e: VariableRef):
    if e.index is not None:
        e.index = restructure(e.index)
    return e


def restructure_composite(e: Composite):
    # Ensure that the transition always has a guard statement.
    if e.guard is None:
        true_primary = Primary(target=True)
        true_primary.produced_statement = True
        e.guard = true_primary
    else:
        e.guard = restructure(e.guard)

    e.assignments = [restructure(v) for v in e.assignments]
    return e


def restructure_assignment(e: Assignment):
    e.left = restructure(e.left)
    e.right = restructure(e.right)
    return e


def restructure_transition(e: Transition):
    # Create copies of the original statements and restructure the copy.
    restructured_statements = []
    for i, s in enumerate(e.statements):
        statement_copy = copy_node(s, dict(), dict())
        restructured_copy = restructure(statement_copy)
        restructured_copy.original_statement = s
        restructured_statements.append(restructured_copy)
    e.statements = restructured_statements

    # Ensure that the transition has a guard statement.
    if len(e.statements) == 0 or isinstance(e.guard, Assignment):
        true_primary = Primary(target=True)
        true_primary.produced_statement = True
        e.statements = [true_primary] + e.statements
    elif isinstance(e.guard, Composite):
        # A composite with a guard that is always true should never be part of the decision structure.
        if e.guard.is_true():
            true_primary = Primary(target=True)
            true_primary.produced_statement = True
            e.statements = [true_primary] + e.statements

    return e


def restructure(e):
    """Preprocess the given expression and add the mandatory structural changes."""
    if isinstance(e, Expression):
        e = restructure_expression(e)
    elif isinstance(e, Primary):
        e = restructure_primary(e)
    elif isinstance(e, VariableRef):
        e = restructure_variable_ref(e)
    elif isinstance(e, Composite):
        e = restructure_composite(e)
    elif isinstance(e, Assignment):
        e = restructure_assignment(e)
    elif isinstance(e, Transition):
        e = restructure_transition(e)
    return e
