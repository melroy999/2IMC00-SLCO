from __future__ import annotations

import operator
from collections import Iterable
from functools import reduce
from typing import TYPE_CHECKING

import objects.ast.models as models

from z3 import z3

from util.smt import operator_mapping, z3_always_holds, z3_never_holds, z3_is_equivalent, z3_is_negation_equivalent

if TYPE_CHECKING:
    from objects.ast.interfaces import SlcoEvaluableNode

# TODO: there are additional associative operators, like xor, that aren't handled as such.
#   - Technically, all operators can be seen as n-ary, as long as the right process is used for resolution.

# Operators that are comparative functions.
comparison_operators = ["!=", "=", "<=", ">=", "<", ">"]


def is_true(e) -> bool:
    """Evaluate whether the statement always holds true. May throw an exception if not a boolean smt statement."""
    byte_values = get_byte_variables(e)
    return z3_always_holds(e.smt, tuple(byte_values))


def is_false(e) -> bool:
    """Evaluate whether the statement never holds true. May throw an exception if not a boolean smt statement."""
    byte_values = get_byte_variables(e)
    return z3_never_holds(e.smt, tuple(byte_values))


def is_equivalent(e, target: SlcoEvaluableNode) -> bool:
    """Evaluate whether the given statements have the same solution space."""
    byte_values = get_byte_variables(e)
    return z3_is_equivalent(e.smt, target.smt, tuple(byte_values))


def is_negation_equivalent(e, target: SlcoEvaluableNode) -> bool:
    """Evaluate whether this statement and the negation of the given statement have the same solution space."""
    byte_values = get_byte_variables(e)
    return z3_is_negation_equivalent(e.smt, target.smt, tuple(byte_values))


def get_byte_variables(e):
    """Get all byte variables used within the statement such that proper byte bounds can be used."""
    exploration_stack = [e]
    byte_variables = set()
    while len(exploration_stack) > 0:
        v = exploration_stack.pop()
        if isinstance(v, models.VariableRef) and v.var.is_byte:
            byte_variables.add(v.var.smt)
        if isinstance(e, Iterable):
            exploration_stack.extend(list(v))
    return byte_variables


def n_ary_to_binary_operations(e: models.Expression, values):
    """Convert an n-ary operation to a chain of binary operations."""
    if len(values) < 2:
        raise Exception("Need at least two values.")
    
    if e.op in comparison_operators:
        # Operators that compare two values cannot be chained in a regular way.
        pairs = []
        for i in range(1, len(values)):
            pairs.append(operator_mapping[e.op](values[i - 1], values[i]))
        return reduce(operator.__and__, pairs)
    elif e.op == "**":
        value = values[-2] ** values[-1]
        for i in range(3, len(values) + 1):
            value = values[-i] ** value
        return value
    else:
        return reduce(operator_mapping[e.op], values)


def composite_to_smt(e: models.Composite):
    return to_smt(e.guard)


def variable_to_smt(e: models.Variable):
    if e.is_boolean:
        if e.is_array:
            return z3.Array(e.name, z3.IntSort(), z3.BoolSort())
        else:
            return z3.Bool(e.name)
    else:
        if e.is_array:
            return z3.Array(e.name, z3.IntSort(), z3.IntSort())
        else:
            return z3.Int(e.name)


def expression_to_smt(e: models.Expression):
    value_smt_statements: list = [to_smt(v) for v in e.values]
    return n_ary_to_binary_operations(e, value_smt_statements)


def primary_to_smt(e: models.Primary):
    if e.value is not None:
        if isinstance(e.value, bool):
            target_value = z3.BoolSort().cast(e.value)
        elif isinstance(e.value, int):
            target_value = z3.IntSort().cast(e.value)
        else:
            raise Exception("Unsupported variable value.")
    elif e.ref is not None:
        target_value = to_smt(e.ref)
    else:
        target_value = to_smt(e.body)

    if e.sign == "-":
        return operator.neg(target_value)
    elif e.sign == "not":
        return z3.Not(target_value)
    else:
        return target_value


def variable_ref_to_smt(e: models.VariableRef):
    var_smt = to_smt(e.var)
    if e.index is None:
        return var_smt
    else:
        return operator.itemgetter(to_smt(e.index))(var_smt)


def to_smt(e):
    """Convert the given statement to an smt object."""
    return conversion_functions[type(e).__name__](e)


# Create a mapping to the conversion functions.
conversion_functions = {
    "Composite": composite_to_smt,
    "Variable": variable_to_smt,
    "Expression": expression_to_smt,
    "Primary": primary_to_smt,
    "VariableRef": variable_ref_to_smt
}
