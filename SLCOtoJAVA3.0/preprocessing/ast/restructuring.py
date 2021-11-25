import logging

from objects.ast.models import Expression, Primary, VariableRef, Composite, Assignment, Transition, StateMachine, \
    SlcoModel, Class
from objects.ast.util import copy_node


def restructure_expression(e: Expression):
    e.values = [restructure(v) for v in e.values]
    if len(e.values) == 1:
        # Skip superfluous expressions that only contain one element.
        logging.debug(f" - Restructuring \"{type(e).__name__}{{op={e.op},values={e.values}}}\" "
                      f"to \"{e.values[0]}\" (#e.values == 1)")
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
        logging.debug(f" - Adding a default true guard to composite \"{e}\" (missing guard)")
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
        logging.debug(f" - Adding a default true guard to transition \"{e}\" (no original guard)")
        true_primary = Primary(target=True)
        true_primary.produced_statement = True
        e.statements = [true_primary] + e.statements
    elif isinstance(e.guard, Composite):
        # A composite with a guard that is always true should never be part of the decision structure.
        if e.guard.is_true():
            logging.debug(f" - Adding a default true valued guard to transition \"{e}\" (composite with true guard)")
            true_primary = Primary(target=True)
            true_primary.produced_statement = True
            e.statements = [true_primary] + e.statements
    return e


def restructure_state_machine(e: StateMachine):
    for t in e.transitions:
        restructure(t)
    return e


def restructure_class(e: Class):
    for sm in e.state_machines:
        restructure(sm)
    return e


def restructure_model(e: SlcoModel):
    for c in e.classes:
        restructure(c)
    return e


def restructure(e):
    """Preprocess the given expression and add the mandatory structural changes."""
    original_e = e
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
    elif isinstance(e, StateMachine):
        e = restructure_state_machine(e)
    elif isinstance(e, Class):
        e = restructure_class(e)
    elif isinstance(e, SlcoModel):
        e = restructure_model(e)

    if str(original_e) != str(e):
        logging.info(f"Restructured {original_e} to {e}")

    return e
