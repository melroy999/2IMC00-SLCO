from __future__ import annotations
from typing import Union

import objects.ast.models as models


def preprocess_expression(e: models.Expression) -> Union[models.Expression, models.Primary]:
    e.values = [preprocess(v) for v in e.values]
    if len(e.values) == 1:
        # Skip superfluous expressions.
        return e.values[0]
    return e


def preprocess_primary(e: models.Primary):
    if e.body is not None:
        e.body = preprocess(e.body)
    elif e.ref is not None:
        e.ref = preprocess(e.ref)
    return e


def preprocess_variable_ref(e: models.VariableRef):
    if e.index is not None:
        e.index = preprocess(e.index)
    return e


def preprocess_composite(e: models.Composite):
    e.guard = preprocess(e.guard)
    e.assignments = [preprocess(v) for v in e.assignments]
    return e


def preprocess_assignment(e: models.Assignment):
    e.left = preprocess(e.left)
    e.right = preprocess(e.right)
    return e


def preprocess_transition(e: models.Transition):
    e.statements = [preprocess(s) for s in e.statements]

    # Remove all statements succeeding a false expression.
    for i, s in enumerate(e.statements):
        if isinstance(s, (models.Expression, models.Primary, models.Composite)):
            if s.is_false():
                e.statements = e.statements[:i] + [models.Primary(target=False)]
                break

    # Remove expressions that always hold true.
    e.statements = [
        s for s in e.statements if not (isinstance(s, (models.Expression, models.Primary)) and s.is_true())
    ]

    # Remove assignments that assign to itself.
    e.statements = [
        s for s in e.statements if not (isinstance(s, models.Assignment) and s.left.is_equivalent(s.right))
    ]

    # Ensure that the transition has a guard.
    if len(e.statements) == 0 or isinstance(e.guard, models.Assignment):
        e.statements = [models.Primary(target=True)] + e.statements
    elif isinstance(e.guard, models.Composite):
        # A composite with a guard that is always true should never be part of the decision structure.
        if e.guard.is_true():
            e.statements = [models.Primary(target=True)] + e.statements

    return e


def preprocess(e: Union[
    models.Expression, models.Primary, models.VariableRef, models.Composite, models.Assignment, models.Transition
]):
    """Preprocess the given expression and add the mandatory structural changes."""
    if isinstance(e, models.Expression):
        e = preprocess_expression(e)
    elif isinstance(e, models.Primary):
        e = preprocess_primary(e)
    elif isinstance(e, models.VariableRef):
        e = preprocess_variable_ref(e)
    elif isinstance(e, models.Composite):
        e = preprocess_composite(e)
    elif isinstance(e, models.Assignment):
        e = preprocess_assignment(e)
    elif isinstance(e, models.Transition):
        e = preprocess_transition(e)
    return e
