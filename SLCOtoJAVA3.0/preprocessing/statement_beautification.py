from __future__ import annotations
from typing import Union

import objects.ast.models as models


# TODO: Add mandatory brackets (left-associated operations with same or lower priority operations at the top).
#   - Potentially remove all efforts to add/retain brackets in the simplification.


# The priorities of the operators, such that superfluous brackets can be removed.
from preprocessing.statement_simplification import associative_operators

operator_priorities = {
    "**": 4,
    "*": 3,
    "/": 3,
    "%": 3,
    "+": 2,
    "-": 2,
    "!=": 1,
    "=": 1,
    "<=": 1,
    ">=": 1,
    "<": 1,
    ">": 1,
    "or": 0,
    "xor": 0,
    "and": 0,
}


# Addition
def beautify_addition(e: models.Expression):
    # Split the factors into positive and negative factors and beautify accordingly.
    positive_factors = []
    negative_factors = []
    for v in e.values:
        # Primaries can be unpacked, since addition and negation have the lowest priority.
        # By construction, addition and subtraction are merged. Hence, it shouldn't matter.
        if isinstance(v, models.Primary) and (v.sign == "-" or v.body is not None):
            if v.sign == "-":
                if v.body is not None:
                    negative_factors.append(v.body)
                else:
                    negative_factors.append(models.Primary(target=v.ref or v.value))
            elif v.body is not None:
                positive_factors.append(v.body)
        else:
            positive_factors.append(v)

    if len(positive_factors) == 0:
        # Wrap addition with a minus sign.
        return models.Primary(sign="-", target=models.Expression("+", negative_factors))
    elif len(negative_factors) == 0:
        # Return as-is.
        return e
    else:
        # Nest the addition as the first factor in a subtraction.
        if len(positive_factors) == 1:
            left = positive_factors[0]
        else:
            left = models.Expression("+", positive_factors)

        # Wrap negative factors in a primary if they contain an addition to preserve subtraction's left-associativity.
        # TODO: This statement isn't needed if the additions in x + -(y + z) are merged appropriately.
        for i, v in enumerate(negative_factors):
            if isinstance(v, models.Expression) and v.op == "+":
                negative_factors[i] = models.Primary(target=v)

        return models.Expression("-", [left] + negative_factors)


def beautify_expression(e: models.Expression) -> Union[models.Expression, models.Primary]:
    # TODO What brackets are needed for a correct result?
    # TODO: Keep this in, since it is important due to the SLCO parser's right-associativity issue.
    values = [beautify(v) for v in e.values]
    # for i, v in enumerate(values):
    #     if isinstance(v, models.Expression):
    #         if operator_priorities[e.op] > operator_priorities[v.op]:
    #             # A bracket is needed if the child nodes contain any operators that are of a lower priority.
    #             values[i] = models.Primary(target=v)
    #         elif operator_priorities[e.op] == operator_priorities[v.op]:
    #             # A bracket is needed at the same priority if it violates the associativity rules.
    #             if e.op == "**":
    #                 # Right-associative.
    #                 if i < len(values):
    #                     values[i] = models.Primary(target=v)
    #             elif e.op not in associative_operators:
    #                 # Left-associative.
    #                 if i > 0:
    #                     values[i] = models.Primary(target=v)
    e.values = values

    if e.op == "+":
        return beautify_addition(e)
    return e


# TODO: the last element in a subtraction doesn't need brackets.

def beautify_primary(e: models.Primary):
    if e.body is not None:
        e.body = beautify(e.body)
        if e.sign == "":
            # TODO: Drop the brackets and only add mandatory ones.
            # return e.body or e


            if isinstance(e.parent, (models.Assignment, models.Composite, models.Transition, models.VariableRef)):
                # Brackets aren't needed when the parent is an assignment, component, transition or variable reference.
                if e.body is not None:
                    return e.body
            elif isinstance(e.parent, models.Expression) and isinstance(e.body, models.Expression):
                # Brackets aren't needed if the priority of the contained element is higher than the one outside.
                if operator_priorities[e.body.op] > operator_priorities[e.parent.op]:
                    return e.body
            elif isinstance(e.body, models.Primary):
                return e.body
    return e


def beautify_variable_ref(e: models.VariableRef):
    if e.index is not None:
        e.index = beautify(e.index)
    return e


def beautify_composite(e: models.Composite):
    e.guard = beautify(e.guard)
    e.assignments = [beautify(v) for v in e.assignments]
    return e


def beautify_assignment(e: models.Assignment):
    e.left = beautify(e.left)
    e.right = beautify(e.right)
    return e


def beautify_transition(e: models.Transition):
    e.statements = [beautify(v) for v in e.statements]

    # Replace composites that do not have assignments with expressions.
    e.statements = [
        s.guard if isinstance(s, models.Composite) and len(s.assignments) == 0 else s for s in e.statements
    ]

    # Replace composites that only have a single assignment and a true guard with an assignment.
    e.statements = [
        s.assignments[0] if isinstance(s, models.Composite) and len(s.assignments) == 1 else s for s in e.statements
    ]

    return e


def beautify(e: Union[
    models.Expression, models.Primary, models.VariableRef, models.Composite, models.Assignment, models.Transition
]):
    """Improve the readability of the given object."""
    if isinstance(e, models.Expression):
        e = beautify_expression(e)
    elif isinstance(e, models.Primary):
        e = beautify_primary(e)
    elif isinstance(e, models.VariableRef):
        e = beautify_variable_ref(e)
    elif isinstance(e, models.Composite):
        e = beautify_composite(e)
    elif isinstance(e, models.Assignment):
        e = beautify_assignment(e)
    elif isinstance(e, models.Transition):
        e = beautify_transition(e)
    return e
