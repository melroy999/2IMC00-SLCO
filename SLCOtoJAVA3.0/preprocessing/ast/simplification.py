import logging
from typing import Union

from objects.ast.interfaces import SlcoEvaluableNode
from objects.ast.models import Expression, Primary, VariableRef, Composite, Assignment, Transition, StateMachine, Class, \
    SlcoModel, Object
from objects.ast.util import copy_node


# A list of all boolean operators that may occur in expressions.
boolean_operators = {"and", "or", "xor", ">", "<", ">=", "<=", "=", "!="}


def simplify_expression(e: Expression):
    if e.op in boolean_operators and e.is_true():
        # Replace the statement with a True primary if it always holds true.
        true_primary = Primary(target=True)
        true_primary.produced_statement = True
        return true_primary
    elif e.op in boolean_operators and e.is_false():
        # Replace the statement with a False primary if it always evaluates to false.
        false_primary = Primary(target=False)
        false_primary.produced_statement = True
        return false_primary
    else:
        e.values = [simplify(v) for v in e.values]
        return e


# Negations of the given operators.
negation_table = {
    "=": "!=",
    "!=": "=",
    ">": "<=",
    "<": ">=",
    ">=": "<",
    "<=": ">",
}


def simplify_primary(e: Primary, recursion=True):
    if e.body is not None:
        if recursion:
            e.body = simplify(e.body)

        if isinstance(e.body, Primary):
            if e.body.value is not None:
                # Elevate the basic value and apply the appropriate sign.
                if e.sign == "":
                    return e.body
                elif e.sign == "-":
                    return Primary(target=-e.body.signed_value)
                else:
                    return Primary(target=not e.body.signed_value)
            else:
                # Swap the signs.
                if e.sign in ["-", "not"] and e.sign == e.body.sign:
                    result = Primary(target=e.body.body or e.body.ref)
                else:
                    result = Primary(sign=e.sign or e.body.sign, target=e.body.body or e.body.ref)
                return simplify_primary(result, recursion=False)
        elif e.body.op in negation_table:
            # Negate operators that can be negated.
            return Expression(negation_table[e.body.op], e.body.values)
    elif e.ref is not None:
        if recursion:
            e.ref = simplify(e.ref)
    return e


def simplify_variable_ref(e: VariableRef):
    if e.index is not None:
        e.index = simplify(e.index)
    return e


def simplify_composite(e: Composite):
    e.guard = simplify(e.guard)
    e.assignments = [simplify(v) for v in e.assignments]

    if e.guard.is_false():
        # A false guarded composite is equivalent to its guard.
        return e.guard
    if e.guard.is_true() and len(e.assignments) == 1:
        # A composite with a true guard and only one assignment can be converted to an assignment.
        return e.assignments[0]
    elif len(e.assignments) == 0:
        # A composite with no assignments is equivalent to its guard.
        return e.guard
    else:
        return e


def simplify_assignment(e: Assignment):
    e.left = simplify(e.left)
    e.right = simplify(e.right)
    return e


def simplify_transition(e: Transition):
    # Create copies of the original statements and simplify the copy.
    restructured_statements = []
    for i, s in enumerate(e.statements):
        statement_copy = copy_node(s, dict(), dict())
        simplified_copy = simplify(statement_copy)
        simplified_copy.original_statement = s
        restructured_statements.append(simplified_copy)
    e.statements = restructured_statements

    # Exclude all statements succeeding a false expression.
    for i, s in enumerate(e.statements):
        if isinstance(s, SlcoEvaluableNode) and s.is_false():
            # Exclude all succeeding statements from rendering, since the code is unreachable.
            for j in range(i + 1, len(e.statements)):
                e.statements[i].exclude_statement = True
            break

    # Exclude expressions that always hold true.
    for s in e.statements:
        if isinstance(s, SlcoEvaluableNode) and s.is_true():
            if isinstance(s, Composite):
                s.guard.exclude_statement = True
            else:
                s.exclude_statement = True

    # Exclude assignments that assign itself.
    for s in e.statements:
        if isinstance(s, Assignment) and s.left.is_equivalent(s.right):
            s.exclude_statement = True

    return e


def assign_missing_default_values(e: Union[StateMachine, Class]) -> None:
    """
    Assign default values to all of the variables if missing.
    """
    for v in e.variables:
        if v.is_array:
            if len(v.def_values) == 0:
                v.def_values = [False if v.is_boolean else 0] * v.type.size
        elif v.def_value is None:
            v.def_value = False if v.is_boolean else 0


def simplify_state_machine(e: StateMachine):
    for t in e.transitions:
        simplify(t)

    # Ensure that all of the state machine variables have an initial value.
    assign_missing_default_values(e)

    return e


def simplify_class(e: Class):
    for sm in e.state_machines:
        simplify(sm)

    # Ensure that all of the class variables have an initial value.
    assign_missing_default_values(e)

    return e


def simplify_object(e: Object):
    # Ensure that all of variables of the objects have initial values.
    value_assignments = {e.left: e.rights if e.right is None else e.right for e in e.assignments}
    e.initial_values = []
    for v in e.type.variables:
        if v in value_assignments:
            e.initial_values.append(value_assignments[v])
        elif v.is_array:
            e.initial_values.append(v.def_values)
        else:
            e.initial_values.append(v.def_value)
    return e


def simplify_model(e: SlcoModel):
    for c in e.classes:
        simplify(c)
    for c in e.objects:
        simplify(c)
    return e


def simplify(e):
    """Simplify the given expression."""
    original_e = e
    if isinstance(e, Expression):
        e = simplify_expression(e)
    elif isinstance(e, Primary):
        e = simplify_primary(e)
    elif isinstance(e, VariableRef):
        e = simplify_variable_ref(e)
    elif isinstance(e, Composite):
        e = simplify_composite(e)
    elif isinstance(e, Assignment):
        e = simplify_assignment(e)
    elif isinstance(e, Transition):
        e = simplify_transition(e)
    elif isinstance(e, StateMachine):
        e = simplify_state_machine(e)
    elif isinstance(e, Class):
        e = simplify_class(e)
    elif isinstance(e, Object):
        e = simplify_object(e)
    elif isinstance(e, SlcoModel):
        e = simplify_model(e)

    if str(original_e) != str(e):
        logging.info(f"Simplified {original_e} to {e}")

    return e
