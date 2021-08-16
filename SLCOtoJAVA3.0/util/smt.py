import operator
from functools import lru_cache

import z3

# Keep a global and reusable instance of the z3 solver.
solver = z3.Solver()
solver.set("timeout", 600)

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


@lru_cache(maxsize=None)
def z3_always_holds(expression, byte_values: tuple) -> bool:
    """Check whether the given expression always holds true."""
    solver.push()
    solver.add(z3.Not(expression))
    for v in byte_values:
        solver.add(z3.And(0 <= v, v < 256))
    result = solver.check()
    solver.pop()

    # The expression always holds true if the negation is satisfiable.
    return result.r == z3.Z3_L_FALSE


@lru_cache(maxsize=None)
def z3_never_holds(expression, byte_values: tuple):
    """Check whether the given expression has no solutions."""
    solver.push()
    solver.add(expression)
    for v in byte_values:
        solver.add(z3.And(0 <= v, v < 256))
    result = solver.check()
    solver.pop()

    # The expression has no solutions if the model is unsatisfiable.
    return result.r == z3.Z3_L_FALSE


@lru_cache(maxsize=None)
def z3_is_equivalent(expression1, expression2, byte_values: tuple):
    """Check whether the given expressions are equivalent."""
    solver.push()
    solver.add(expression1 != expression2)
    for v in byte_values:
        solver.add(z3.And(0 <= v, v < 256))
    result = solver.check()
    solver.pop()

    # The expression has no solutions if the model is unsatisfiable.
    return result.r == z3.Z3_L_FALSE


@lru_cache(maxsize=None)
def z3_is_negation_equivalent(expression1, expression2, byte_values: tuple):
    """Check whether the negation of the second expression is equivalent to the first expression."""
    solver.push()
    try:
        solver.add(expression1 != z3.Not(expression2))
    except z3.z3types.Z3Exception:
        solver.add(expression1 != -expression2)

    for v in byte_values:
        solver.add(z3.And(0 <= v, v < 256))
    result = solver.check()
    solver.pop()

    # The expression has no solutions if the model is unsatisfiable.
    return result.r == z3.Z3_L_FALSE


def clear_smt_cache():
    z3_always_holds.cache_clear()
    z3_never_holds.cache_clear()
    z3_is_equivalent.cache_clear()
    z3_is_negation_equivalent.cache_clear()
