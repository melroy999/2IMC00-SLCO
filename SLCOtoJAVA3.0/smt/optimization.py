from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, Set

from z3 import z3

from objects.ast.models import DecisionNode, GuardNode

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


def create_is_true_table(transitions, alias_variables) -> None:
    """Create a truth table for the given list of transitions that denotes if a guard is always true."""
    # Create the truth table.
    for t in transitions:
        alias_variables[f"it{t.id}"] = z3.Bool(f"it{t.id}")

    for t in transitions:
        # Add a new negated equality relation to the solver.
        e = t.guard.smt

        # Existential quantifiers are not supported, so do the check separately.
        s.push()
        s.add(z3.Not(e))
        result = s.check().r == z3.Z3_L_FALSE
        s.pop()

        # Add the constant to the solver.
        s.add(alias_variables[f"it{t.id}"] == result)


def create_decision_groupings(transitions: List[Transition]) -> Union[DecisionNode, Transition]:
    """
    Use z3 optimization to create a minimally sized collections of groups of transitions in which the guard statements'
    active regions do not overlap.
    """
    if len(transitions) == 1:
        # Return the transition.
        return transitions[0]

    # Give all transitions an unique identity.
    for i, t in enumerate(transitions):
        t.id = i

    # Filter out all transitions that are invariably true.
    trivially_satisfiable_transitions = []
    remaining_transitions = []
    for t in transitions:
        if t.guard.is_true():
            trivially_satisfiable_transitions.append(t)
        else:
            remaining_transitions.append(t)

    # Save the current state of the solver.
    s.push()

    # Create variables for each transition that will indicate whether they are part of the group or not.
    alias_variables = {f"g{t.id}": z3.Int(f"g{t.id}") for t in remaining_transitions}

    # Create the appropriate truth tables for the target transitions.
    create_and_truth_table(remaining_transitions, alias_variables)
    create_is_equal_table(remaining_transitions, alias_variables)

    # Ensure that the part of the group variable of each transition can only be zero or one.
    for t in remaining_transitions:
        v = alias_variables[f"g{t.id}"]
        s.add(z3.And(v >= 0, v < 2))

    # Calculate truth matrices for the transitions that still need to be assigned to a group.
    non_deterministic_choices: List[Union[DecisionNode, Transition, GuardNode]] = trivially_satisfiable_transitions
    while len(remaining_transitions) > 0:
        # Save the current state of the solver.
        s.push()

        # Create the rules needed to construct valid groupings.
        for t1 in remaining_transitions:
            v = alias_variables[f"g{t1.id}"]
            inner_or = z3.Or([
                z3.And(
                    alias_variables[f"g{t2.id}"] == 1,
                    alias_variables[f"and{t1.id}_{t2.id}"],
                    z3.Not(alias_variables[f"ieq{t1.id}_{t2.id}"])
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

        # Extract the deterministic choices.
        # Moreover, merge transitions with the exact same solution space into a non-deterministic decision.
        deterministic_choices: List[Union[Transition, GuardNode, DecisionNode]] = []
        processed_transition_ids: Set[int] = set()
        for t in grouped_transitions:
            # Skip transitions that have been encountered already.
            if t.id in processed_transition_ids:
                continue

            # Find all transitions that have the same solution space.
            equivalent_transitions = [
                t2 for t2 in grouped_transitions if model.evaluate(
                    alias_variables[f"ieq{t.id}_{t2.id}"], model_completion=True
                )
            ]

            # Wrap the transitions in a non-deterministic decision node if needed.
            if len(equivalent_transitions) == 1:
                deterministic_choices.append(t)
                processed_transition_ids.add(t.id)
            else:
                # Create a nested group and mark all contained transitions as processed.
                deterministic_choices.append(GuardNode(t.guard, DecisionNode(equivalent_transitions, False)))
                processed_transition_ids.update([t.id for t in equivalent_transitions])

        # Create a deterministic decision node for the current grouping and add it to the list of groupings.
        if len(deterministic_choices) == 1:
            non_deterministic_choices.append(deterministic_choices[0])
        else:
            non_deterministic_choices.append(DecisionNode(deterministic_choices, True))

        # Remove the transitions from the to be processed list.
        for t in grouped_transitions:
            remaining_transitions.remove(t)

        # Restore the state of the solver.
        s.pop()

    # Restore the state of the solver.
    s.pop()

    # Create and return a non-deterministic decision node for the given decisions.
    return DecisionNode(non_deterministic_choices, False)

# def create_minimal_groupings_of_non_overlapping_active_regions(
#         transitions: List[Transition]
# ) -> Union[DecisionNode, Transition]:
#     """
#     Use z3 optimization to create a minimally sized list of groups of transitions in which the guard statements'
#     active regions do not overlap.
#     """
#     # TODO: This code performs quite slow when using larger models. Does the old approach fare better?
#     if len(transitions) == 1:
#         # Return the transition as a single element non-deterministic decision node.
#         return DecisionNode(transitions, False)
#
#     # Give all of the transitions within the list an unique ID.
#     # This id can be used to find the original ordering.
#     for i, t in enumerate(transitions):
#         t.id = i
#
#     # Save the current state of the solver.
#     s.push()
#
#     # Create variables for each transition that will indicate the number of the group they are in.
#     alias_variables = {f"g{t.id}": z3.Int(f"g{t.id}") for t in transitions}
#
#     # Create the appropriate truth tables for the target transitions.
#     create_and_truth_table(transitions, alias_variables)
#     create_is_equal_table(transitions, alias_variables)
#     create_is_true_table(transitions, alias_variables)
#
#     # Put bounds on the group variables such that they are always within a range between zero and the list's size.
#     for t in transitions:
#         v = alias_variables[f"g{t.id}"]
#         s.add(z3.And(v >= 0, v < len(transitions)))
#
#     # Ensure that a transition can only be assigned to a group if it does not overlap with others in the same group.
#     # However, make the following exception: allow transitions to be in the same group if they are exactly equal.
#     for t1 in transitions:
#         v = alias_variables[f"g{t1.id}"]
#         for k in range(len(transitions)):
#             inner_or = z3.Or([
#                 z3.And(
#                     alias_variables[f"g{t2.id}"] == k,
#                     alias_variables[f"and{t1.id}_{t2.id}"],
#                     z3.Not(alias_variables[f"ieq{t1.id}_{t2.id}"])
#                 ) for t2 in transitions if t1.id != t2.id
#             ])
#             s.add(z3.Implies(v == k, z3.Not(inner_or)))
#
#             # If the statement is trivially satisfiable (always true), then it should not share a group with others.
#             s.add(z3.Implies(
#                 z3.And(v == k, alias_variables[f"it{t1.id}"]),
#                 z3.Not(z3.Or([
#                     alias_variables[f"g{t2.id}"] == k for t2 in transitions if t1.id != t2.id
#                 ])))
#             )
#
#     # Minimize the sum of all groups such that the least number of groups are generated with maximum size.
#     s.minimize(sum(alias_variables[f"g{t.id}"] for t in transitions))
#
#     # Get the model solution and extract the selected transitions.
#     result, model = s.check(), s.model()
#     if result.r == z3.Z3_L_UNDEF:
#         print(result, model)
#         raise Exception("Unknown result.")
#     if result.r == z3.Z3_L_FALSE:
#         print(result, model)
#         raise Exception("Unsatisfiable result.")
#
#     # Find the groupings. Elements within the same group assigned should be added to the same group.
#     decisions: List[DecisionNode, Transition] = []
#     for k, _ in enumerate(transitions):
#         target_transitions: List[Transition] = []
#         for t in transitions:
#             if model.evaluate(alias_variables[f"g{t.id}"], model_completion=True) == k:
#                 target_transitions.append(t)
#
#         if len(target_transitions) > 0:
#             current_decisions: List[Union[Transition, GuardNode, DecisionNode]] = []
#             processed_transition_ids: Set[int] = set()
#             for t in target_transitions:
#                 if t.id in processed_transition_ids:
#                     continue
#
#                 # Find all transitions that have the same solution space.
#                 equal_transitions = [
#                     t2 for t2 in target_transitions if model.evaluate(
#                         alias_variables[f"ieq{t.id}_{t2.id}"], model_completion=True
#                     )
#                 ]
#
#                 # Wrap the transitions in a non-deterministic decision node if needed.
#                 if len(equal_transitions) == 1:
#                     current_decisions.append(t)
#                     processed_transition_ids.add(t.id)
#                 else:
#                     # Create a nested group and mark all contained transitions as processed.
#                     current_decisions.append(GuardNode(t.guard, DecisionNode(equal_transitions, False)))
#                     processed_transition_ids.update([t.id for t in equal_transitions])
#
#             # Create a deterministic decision node for the current grouping and add it to the list of groupings.
#             if len(current_decisions) == 1:
#                 decisions.append(current_decisions[0])
#             else:
#                 decisions.append(DecisionNode(current_decisions, True))
#
#     # Restore the state of the solver.
#     s.pop()
#
#     # Create and return a non-deterministic decision node for the given decisions.
#     return DecisionNode(decisions, False)
