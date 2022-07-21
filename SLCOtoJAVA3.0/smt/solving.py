from __future__ import annotations

from cachetools import cached
from typing import TYPE_CHECKING, Iterable
from smt.util import z3_always_holds, z3_never_holds, z3_is_equivalent, z3_is_negation_equivalent, z3_exists_lte

if TYPE_CHECKING:
    from objects.ast.interfaces import SlcoEvaluableNode


@cached(cache={}, key=lambda e: e.id)
def is_true(e) -> bool:
    """Evaluate whether the statement always holds true. May throw an exception if not a boolean smt statement."""
    byte_values = get_byte_variables(e)
    return z3_always_holds(e.smt, byte_values)


@cached(cache={}, key=lambda e: e.id)
def is_false(e) -> bool:
    """Evaluate whether the statement never holds true. May throw an exception if not a boolean smt statement."""
    byte_values = get_byte_variables(e)
    return z3_never_holds(e.smt, byte_values)


@cached(cache={}, key=lambda e1, e2: (e1.id, e2.id))
def is_equivalent(e, target: SlcoEvaluableNode) -> bool:
    """Evaluate whether the given statements have the same solution space."""
    byte_values = get_byte_variables(e)
    return z3_is_equivalent(e.smt, target.smt, byte_values)


@cached(cache={}, key=lambda e1, e2: (e1.id, e2.id))
def is_negation_equivalent(e, target: SlcoEvaluableNode) -> bool:
    """Evaluate whether this statement and the negation of the given statement have the same solution space."""
    byte_values = get_byte_variables(e)
    return z3_is_negation_equivalent(e.smt, target.smt, byte_values)


def exists_lte(e, target: SlcoEvaluableNode) -> bool:
    """Evaluate whether there exists a solution in which e is less than or equal to the target."""
    byte_values = get_byte_variables(e)
    return z3_exists_lte(e.smt, target.smt, byte_values)


def get_byte_variables(e):
    """Get all byte variables used within the statement such that proper byte bounds can be used."""
    exploration_stack = [e]
    byte_variables = set()
    while len(exploration_stack) > 0:
        v = exploration_stack.pop()
        if type(v).__name__ == "VariableRef" and v.var.is_byte:
            byte_variables.add(v.smt)
        if isinstance(e, Iterable):
            exploration_stack.extend(list(v))
    return byte_variables
