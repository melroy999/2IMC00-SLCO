from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

from z3 import z3

if TYPE_CHECKING:
    from objects.ast.models import Expression, Composite, Variable, Primary, VariableRef

# Operators that are comparative functions.
comparison_operators = ["!=", "=", "<=", ">=", "<", ">"]

# Map every operator to its implementation to avoid calling eval.
operator_mapping = {
    ">": operator.__gt__,
    "<": operator.__lt__,
    ">=": operator.__ge__,
    "<=": operator.__le__,
    "=": operator.__eq__,
    "!=": operator.__ne__,
    "<>": operator.__ne__,
    "+": operator.__add__,
    "-": operator.__sub__,
    "*": operator.__mul__,
    "**": operator.__pow__,
    "%": operator.__mod__,
    # It is safe to use truediv for integer divisions in SMT, since it defaults to integer divisions.
    "/": operator.__truediv__,
    "or": z3.Or,
    "||": z3.Or,
    "and": z3.And,
    "&&": z3.And,
    "xor": z3.Xor,
    "": lambda v: v,
}


def n_ary_to_binary_operations(e: Expression, values):
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


def composite_to_smt(e: Composite):
    """Convert the given composite to an smt object."""
    return to_smt(e.guard)


def variable_to_smt(e: Variable):
    """Convert the given variable to an smt object."""
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


def expression_to_smt(e: Expression):
    """Convert the given expression to an smt object."""
    value_smt_statements: list = [to_smt(v) for v in e.values]
    return n_ary_to_binary_operations(e, value_smt_statements)


def primary_to_smt(e: Primary):
    """Convert the given primary to an smt object."""
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


def variable_ref_to_smt(e: VariableRef):
    """Convert the given variable reference to an smt object."""
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
