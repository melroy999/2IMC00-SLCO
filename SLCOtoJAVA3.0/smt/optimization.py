from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, Set

from z3 import z3

from objects.ast.models import DecisionNode, GuardNode

if TYPE_CHECKING:
    from objects.ast.models import Transition

# Create a solver object for optimization problems.
s = z3.Optimize()


def create_and_truth_table(transitions, alias_variables) -> None:
    """Create a truth table for the AND operation between the given list of transitions"""
    # Create the truth table.
    for i, _ in enumerate(transitions):
        for j, _ in enumerate(transitions):
            alias_variables[f"and{i}_{j}"] = z3.Bool(f"and{i}_{j}")

    for i, t1 in enumerate(transitions):
        for j, t2 in enumerate(transitions):
            if j > i:
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
            s.add(alias_variables[f"and{i}_{j}"] == result)
            if j < i:
                s.add(alias_variables[f"and{i}_{j}"] == alias_variables[f"and{i}_{j}"])


def create_is_equal_table(transitions, alias_variables) -> None:
    """Create a truth table for the negated equality operation between the given list of transitions"""
    # Create the truth table.
    for i, _ in enumerate(transitions):
        for j, _ in enumerate(transitions):
            alias_variables[f"ieq{i}_{j}"] = z3.Bool(f"ieq{i}_{j}")

    for i, t1 in enumerate(transitions):
        for j, t2 in enumerate(transitions):
            if j > i:
                break

            # Add a new AND relation to the solver.
            e1 = t1.guard.smt
            e2 = t2.guard.smt

            # Existential quantifiers are not supported, so do the check separately.
            s.push()
            s.add(z3.Not(e1 == e2))
            result = s.check().r == z3.Z3_L_FALSE
            s.pop()

            # Add the constant to the solver.
            s.add(alias_variables[f"ieq{i}_{j}"] == result)
            if j < i:
                s.add(alias_variables[f"ieq{i}_{j}"] == alias_variables[f"ieq{i}_{j}"])


def create_minimal_groupings_of_non_overlapping_active_regions(
        transitions: List[Transition]
) -> Union[DecisionNode, Transition]:
    """
    Use z3 optimization to create a minimally sized collections of groups of transitions in which the guard statements'
    active regions do not overlap.
    """
    # TODO: use the decision structure objects instead.
    if len(transitions) == 1:
        # Return the transition as a single element non-deterministic decision node.
        return DecisionNode(transitions, False)

    # Give all of the transitions within the list an unique ID.
    # This id can be used to find the original ordering.
    for i, t in enumerate(transitions):
        t.id = i

    # Save the current state of the solver.
    s.push()

    # Create variables for each transition that will indicate the number of the group they are in.
    alias_variables = {f"g{t.id}": z3.Int(f"g{t.id}") for t in transitions}

    # Create the appropriate truth tables for the target transitions.
    create_and_truth_table(transitions, alias_variables)
    create_is_equal_table(transitions, alias_variables)

    # Put bounds on the group variables such that they are always within a range between zero and the list's size.
    for t in transitions:
        v = alias_variables[f"g{t.id}"]
        s.add(z3.And(v >= 0, v < len(transitions)))

    # Ensure that a transition can only be assigned to a group if it does not overlap with others in the same group.
    # However, make the following exception: allow transitions to be in the same group if they are exactly equal.
    for t1 in transitions:
        v = alias_variables[f"g{t1.id}"]
        for k in range(len(transitions)):
            inner_or = z3.Or([
                z3.And(
                    alias_variables[f"g{t2.id}"] == k,
                    alias_variables[f"and{t1.id}_{t2.id}"],
                    z3.Not(alias_variables[f"ieq{t1.id}_{t2.id}"])
                ) for t2 in transitions if t1.id != t2.id
            ])
            s.add(z3.Implies(v == k, z3.Not(inner_or)))

    # Minimize the sum of all groups such that the least number of groups are generated with maximum size.
    s.minimize(sum(alias_variables[f"g{t.id}"] for t in transitions))

    # Get the model solution and extract the selected transitions.
    result, model = s.check(), s.model()
    if result.r == z3.Z3_L_UNDEF:
        print(result, model)
        raise Exception("Unknown result.")
    if result.r == z3.Z3_L_FALSE:
        print(result, model)
        raise Exception("Unsatisfiable result.")

    # Find the groupings. Elements within the same group assigned should be added to the same group.
    decisions: List[DecisionNode, Transition] = []
    for k, _ in enumerate(transitions):
        target_transitions: List[Transition] = []
        for t in transitions:
            if model.evaluate(alias_variables[f"g{t.id}"], model_completion=True) == k:
                target_transitions.append(t)

        if len(target_transitions) > 0:
            current_decisions: List[Union[Transition, GuardNode, DecisionNode]] = []
            processed_transition_ids: Set[int] = set()
            while len(target_transitions) > 0:
                # Take the first element.
                t = target_transitions.pop()
                if t.id in processed_transition_ids:
                    continue

                # Find all transitions that have the same solution space.
                equal_transitions = [
                    t2 for t2 in transitions if model.evaluate(
                        alias_variables[f"ieq{t.id}_{t2.id}"], model_completion=True
                    )
                ]

                # Wrap the transitions in a non-deterministic decision node if needed.
                if len(equal_transitions) == 1:
                    current_decisions.append(t)
                    processed_transition_ids.add(t.id)
                else:
                    # Create a nested group and mark all contained transitions as processed.
                    current_decisions.append(GuardNode(t.guard, DecisionNode(equal_transitions, False)))
                    processed_transition_ids.update([t.id for t in equal_transitions])

            # Create a deterministic decision node for the current grouping and add it to the list of groupings.
            if len(current_decisions) == 1:
                decisions.append(current_decisions[0])
            else:
                decisions.append(DecisionNode(current_decisions, True))

    # Restore the state of the solver.
    s.pop()

    # Create and return a non-deterministic decision node for the given decisions.
    return DecisionNode(decisions, False)
