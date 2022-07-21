import operator
from functools import lru_cache

import z3

# Keep a global and reusable instance of the z3 solver.
solver = z3.Solver()
solver.set("timeout", 600)


def z3_always_holds(expression, byte_values: set) -> bool:
    """Check whether the given expression always holds true."""
    solver.push()
    solver.add(z3.Not(expression))
    for v in byte_values:
        solver.add(z3.And(0 <= v, v < 256))
    result = solver.check()
    solver.pop()

    # The expression always holds true if the negation is satisfiable.
    return result.r == z3.Z3_L_FALSE


def z3_never_holds(expression, byte_values: set) -> bool:
    """Check whether the given expression has no solutions."""
    solver.push()
    solver.add(expression)
    for v in byte_values:
        solver.add(z3.And(0 <= v, v < 256))
    result = solver.check()
    solver.pop()

    # The expression has no solutions if the model is unsatisfiable.
    return result.r == z3.Z3_L_FALSE


def z3_is_equivalent(expression1, expression2, byte_values: set) -> bool:
    """Check whether the given expressions are equivalent."""
    solver.push()
    solver.add(expression1 != expression2)
    for v in byte_values:
        solver.add(z3.And(0 <= v, v < 256))
    result = solver.check()
    solver.pop()

    # The expressions are equivalent if no instance can be found for which they are not equal.
    return result.r == z3.Z3_L_FALSE


def z3_is_negation_equivalent(expression1, expression2, byte_values: set) -> bool:
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

    # The expressions are equivalent if no instance can be found for which they are not equal.
    return result.r == z3.Z3_L_FALSE


def z3_exists_lte(expression1, expression2, byte_values: set) -> bool:
    """Check whether the first expression has a solution for which it less than or equal to the second."""
    solver.push()
    solver.add(expression1 <= expression2)
    for v in byte_values:
        solver.add(z3.And(0 <= v, v < 256))
    result = solver.check()
    solver.pop()

    # A solution existing implies that expression one may be less than equal to expression two.
    return result.r == z3.Z3_L_TRUE
