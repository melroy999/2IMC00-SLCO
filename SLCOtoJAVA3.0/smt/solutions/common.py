from typing import List, Dict

from z3 import z3

# Create solver objects for optimization problems.
from objects.ast.models import Transition


def create_and_truth_table(transitions, alias_variables, s) -> None:
    """Create a truth table for the AND operation between the given list of transitions."""
    # Create the truth table.
    for t1 in transitions:
        for t2 in transitions:
            alias_variables[f"and{t1.id}_{t2.id}"] = z3.Bool(f"and{t1.id}_{t2.id}")

    for t1 in transitions:
        for t2 in transitions:
            if t2.id > t1.id:
                break

            # Add a new AND relation to the solver.
            e1 = t1.guard.smt
            e2 = t2.guard.smt

            # Existential quantifiers are not supported, so do the check separately.
            s.push()
            s.add(z3.And(e1, e2))
            # noinspection DuplicatedCode
            result = s.check().r == z3.Z3_L_TRUE
            s.pop()

            # Add the constant to the solver.
            s.add(alias_variables[f"and{t1.id}_{t2.id}"] == result)
            if t2.id < t1.id:
                s.add(alias_variables[f"and{t2.id}_{t1.id}"] == alias_variables[f"and{t1.id}_{t2.id}"])


def create_is_equal_table(transitions, alias_variables, s) -> None:
    """Create a truth table for the negated equality operation between the given list of transitions."""
    # Create the truth table.
    for t1 in transitions:
        for t2 in transitions:
            alias_variables[f"ieq{t1.id}_{t2.id}"] = z3.Bool(f"ieq{t1.id}_{t2.id}")

    for t1 in transitions:
        for t2 in transitions:
            if t2.id > t1.id:
                break

            # Add a new negated equality relation to the solver.
            e1 = t1.guard.smt
            e2 = t2.guard.smt

            # Existential quantifiers are not supported, so do the check separately.
            s.push()
            s.add(z3.Not(e1 == e2))
            # noinspection DuplicatedCode
            result = s.check().r == z3.Z3_L_FALSE
            s.pop()

            # Add the constant to the solver.
            s.add(alias_variables[f"ieq{t1.id}_{t2.id}"] == result)
            if t2.id < t1.id:
                s.add(alias_variables[f"ieq{t2.id}_{t1.id}"] == alias_variables[f"ieq{t1.id}_{t2.id}"])


def create_contains_table(transitions, alias_variables, s) -> None:
    """Create a truth table for the subset relations between transitions, being true if one contains the other."""
    # Create the truth table.
    for t1 in transitions:
        for t2 in transitions:
            alias_variables[f"contains{t1.id}_{t2.id}"] = z3.Bool(f"contains{t1.id}_{t2.id}")

    for t1 in transitions:
        for t2 in transitions:
            # Add a new negated equality relation to the solver.
            e1 = t1.guard.smt
            e2 = t2.guard.smt

            # Existential quantifiers are not supported, so do the check separately.
            s.push()
            s.add(z3.Not(z3.Implies(e2, e1)))
            # noinspection DuplicatedCode
            result = s.check().r == z3.Z3_L_FALSE
            s.pop()

            # Add the constant to the solver.
            s.add(alias_variables[f"contains{t1.id}_{t2.id}"] == result)


def create_smt_support_variables(
        remaining_transitions: List[Transition], s, include_and=True, include_equal=True, include_contains=True
) -> Dict[str, z3.ArithRef]:
    """Create a dictionary of SMT variables that are used in all implemented solutions."""
    # Create variables for each transition that will indicate the number of the group they are in.
    alias_variables = {f"g{t.id}": z3.Int(f"g{t.id}") for t in remaining_transitions}

    # Create support variables for the priorities assigned to the transitions.
    for t in remaining_transitions:
        v = alias_variables[f"p{t.id}"] = z3.Int(f"p{t.id}")
        s.add(v == t.priority)

    # Create the appropriate truth tables for the target transitions.
    if include_and:
        create_and_truth_table(remaining_transitions, alias_variables, s)
    if include_equal:
        create_is_equal_table(remaining_transitions, alias_variables, s)
    if include_contains:
        create_contains_table(remaining_transitions, alias_variables, s)

    return alias_variables
