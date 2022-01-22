from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, Set

from z3 import z3

import settings
from objects.ast.models import DecisionNode

if TYPE_CHECKING:
    from objects.ast.models import Transition

# Create a solver object for optimization problems.
s = z3.Optimize()


def create_and_truth_table(transitions, alias_variables) -> None:
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
            result = s.check().r == z3.Z3_L_TRUE
            s.pop()

            # Add the constant to the solver.
            s.add(alias_variables[f"and{t1.id}_{t2.id}"] == result)
            if t2.id < t1.id:
                s.add(alias_variables[f"and{t2.id}_{t1.id}"] == alias_variables[f"and{t1.id}_{t2.id}"])


def create_is_equal_table(transitions, alias_variables) -> None:
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
            result = s.check().r == z3.Z3_L_FALSE
            s.pop()

            # Add the constant to the solver.
            s.add(alias_variables[f"ieq{t1.id}_{t2.id}"] == result)
            if t2.id < t1.id:
                s.add(alias_variables[f"ieq{t2.id}_{t1.id}"] == alias_variables[f"ieq{t1.id}_{t2.id}"])


def create_decision_groupings(transitions: List[Transition]) -> DecisionNode:
    """
    Use z3 optimization to create a minimally sized collections of groups of transitions in which the guard statements'
    active regions do not overlap.
    """
    # Give all transitions an unique identity.
    for i, t in enumerate(transitions):
        t.id = i

    # Filter out all transitions that are invariably true.
    trivially_satisfiable_transitions = []
    remaining_transitions = []
    excluded_transitions = []
    for t in transitions:
        if t.guard.is_true():
            trivially_satisfiable_transitions.append(t)
        elif t.guard.is_false():
            excluded_transitions.append(t)
        else:
            remaining_transitions.append(t)

    # Save the current state of the solver.
    s.push()

    # Create variables for each transition that will indicate whether they are part of the group or not.
    alias_variables = {f"g{t.id}": z3.Int(f"g{t.id}") for t in remaining_transitions}

    # Create support variables for the priorities assigned to the transitions.
    for t in remaining_transitions:
        v = alias_variables[f"p{t.id}"] = z3.Int(f"p{t.id}")
        s.add(v == t.priority)

    # Create the appropriate truth tables for the target transitions.
    create_and_truth_table(remaining_transitions, alias_variables)
    create_is_equal_table(remaining_transitions, alias_variables)

    # Ensure that the part of the group variable of each transition can only be zero or one.
    for t in remaining_transitions:
        v = alias_variables[f"g{t.id}"]
        s.add(z3.And(v >= 0, v < 2))

    # Calculate truth matrices for the transitions that still need to be assigned to a group.
    non_deterministic_choices: List[Union[DecisionNode, Transition]] = []
    while len(remaining_transitions) > 0:
        # Save the current state of the solver.
        s.push()

        # Create the rules needed to construct valid groupings.
        for t1 in remaining_transitions:
            v = alias_variables[f"g{t1.id}"]

            # It must hold that the transition has no overlap with any of the members in the same group.
            inner_or = z3.Or([
                z3.And(
                    alias_variables[f"g{t2.id}"] == 1,
                    alias_variables[f"and{t1.id}_{t2.id}"],
                    z3.Not(alias_variables[f"ieq{t1.id}_{t2.id}"])
                ) for t2 in remaining_transitions if t1.id != t2.id
            ])
            s.add(z3.Implies(v == 1, z3.Not(inner_or)))

            # It must hold that the priority of each member in the group is the same.
            inner_or = z3.Or([
                z3.And(
                    alias_variables[f"g{t2.id}"] == 1,
                    alias_variables[f"p{t2.id}"] != alias_variables[f"p{t1.id}"]
                ) for t2 in remaining_transitions if t1.id != t2.id
            ])
            s.add(z3.Implies(v == 1, z3.Not(inner_or)))

        # Maximize the size of the group.
        s.maximize(sum(alias_variables[f"g{t.id}"] for t in remaining_transitions))

        # Get the model solution and extract the selected transitions.
        result, model = s.check(), s.model()
        if result.r == z3.Z3_L_UNDEF:
            print(result, model)
            raise Exception("Unknown result.")
        if result.r == z3.Z3_L_FALSE:
            print(result, model)
            raise Exception("Unsatisfiable result.")

        # Find the target transitions that are part of the deterministic group and process them accordingly.
        grouped_transitions: List[Transition] = [
            t for t in remaining_transitions if model.evaluate(alias_variables[f"g{t.id}"], model_completion=True) == 1
        ]

        # Remove the transitions from the to be processed list.
        for t in grouped_transitions:
            remaining_transitions.remove(t)

        # Ensure that duplicates in the group are handled appropriately.
        distinct_guard_groups: List[List[Transition]] = []
        for t in grouped_transitions:
            d: List[Transition]
            # Warning: syntax is correct. The else will only be executed if the loop isn't stopped prematurely.
            for d in distinct_guard_groups:
                if model.evaluate(alias_variables[f"ieq{t.id}_{d[0].id}"], model_completion=True):
                    d.append(t)
                    break
            else:
                distinct_guard_groups.append([t])

        # Sort the guard groups by priority and index and make non-deterministic groups if appropriate.
        deterministic_choices: List[Union[DecisionNode, Transition]] = []
        for d in distinct_guard_groups:
            d.sort(key=lambda x: (x.priority, x.id))
            if len(d) > 1:
                if not settings.non_determinism:
                    # Given that all transitions have the same guard, only the first will be reachable. Exclude others.
                    deterministic_choices.append(DecisionNode(False, d[:1], d[1:]))
                else:
                    # Create a non-deterministic block.
                    deterministic_choices.append(DecisionNode(False, d, []))
            else:
                deterministic_choices.append(d[0])

        # Create a deterministic decision node for the current grouping and add it to the list of groupings.
        if len(deterministic_choices) == 1:
            non_deterministic_choices.append(deterministic_choices[0])
        else:
            # Sort the decisions based on the priority.
            deterministic_choices.sort(key=lambda x: (x.priority, x.id))
            non_deterministic_choices.append(DecisionNode(True, deterministic_choices, []))

        # Restore the state of the solver.
        s.pop()

    # Restore the state of the solver.
    s.pop()

    # TODO: decisions following a true expression in the outer non deterministic group are superfluous if sequential.
    # TODO: potentially have transitions with location sensitive locking targets be earlier in the list?
    #   - The order of statements can have a large effect on the performance of the locking structure.
    #   - This would need to be done before the full locking structure is generated.

    # Sort the decisions based on the priority.
    non_deterministic_choices += trivially_satisfiable_transitions
    non_deterministic_choices.sort(key=lambda x: (x.priority, x.id))

    # Create and return a non-deterministic decision node for the given decisions.
    return DecisionNode(False, non_deterministic_choices, excluded_transitions)
