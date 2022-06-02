from __future__ import annotations

from typing import List, Union, Dict

from z3 import z3

from objects.ast.models import Transition, DecisionNode
from smt.solutions.common import create_smt_support_variables


# Create solver object for optimization problems.
s = z3.Optimize()


def construct_deterministic_node(
        transitions: List[Transition],
        alias_variables: Dict[str, z3.ArithRef],
        and_table: Dict[int, Dict[int, bool]],
        contains_table: Dict[int, Dict[int, bool]]
) -> Union[DecisionNode, Transition]:
    """Select non-deterministic groups within the given list of transitions."""
    # The list of transitions may have remaining overlaps--note however that all overlaps will be contains relations.
    # Hence, regroup the transitions, based on the contains attribute.
    decision_groups: List[List[Transition]] = []

    # Count the number of contain relations the transition has.
    number_of_contains = {t1: len([t2 for t2 in transitions if contains_table[t1.id][t2.id]]) for t1 in transitions}

    # Visit transitions with the most contained elements first--this simplifies the algorithm.
    transitions.sort(reverse=True, key=lambda t1: number_of_contains[t1])
    for t in transitions:
        for d in decision_groups:
            if contains_table[d[0].id][t.id]:
                d.append(t)
                break
        else:
            decision_groups.append([t])

    # All decision groups will become nested non-deterministic nodes.
    deterministic_choices: List[Union[DecisionNode, Transition]] = []
    for g in decision_groups:
        if len(g) > 1:
            # Find the encompassing statement, which may act as the decision node's guard statement.
            guard_statement = g[0]

            # Sort such that the transition with the highest priority and lowest id is first in the list.
            g.sort(key=lambda x: (x.priority, x.id))

            # Create a non-deterministic block.
            decision_node = construct_non_deterministic_node(g, alias_variables, and_table, contains_table)
            decision_node.guard_statement = guard_statement
            deterministic_choices.append(decision_node)
        else:
            deterministic_choices.append(g[0])

    # Sort the deterministic choices and create a deterministic node.
    deterministic_choices.sort(key=lambda x: (x.priority, x.id))
    deterministic_node = DecisionNode(True, deterministic_choices, [])
    # TODO: lock ordering optimization.
    return deterministic_node


def remove_conflicting_transitions(overlap_table, transitions):
    """Remove transitions from the transition list that overlap with every other transition."""
    conflicting_transitions = []
    for t1 in transitions:
        if all(overlap_table[t1.id][t2.id] for t2 in transitions):
            conflicting_transitions.append(t1)
    for t in conflicting_transitions:
        transitions.remove(t)

    # Return the list of conflicting transitions.
    return conflicting_transitions


def get_non_deterministic_node_groups(
        transitions: List[Transition],
        alias_variables: Dict[str, z3.ArithRef]
) -> List[List[Union[DecisionNode, Transition]]]:
    """Convert the list of transitions to groups that are to be contained within a non-deterministic decision node."""
    # Assign the given transitions to groups that internally display deterministic behavior.
    deterministic_transition_groups: List[List[Union[DecisionNode, Transition]]] = []

    # Select groups until no transitions remain.
    while len(transitions) > 0:
        # Save the current state of the solver.
        s.push()

        # Create the rules needed to construct valid groupings.
        for t1 in transitions:
            v = alias_variables[f"g{t1.id}"]

            # It must hold that the transition has no overlap with any of the members in the same group.
            inner_or = z3.Or([
                z3.And(
                    alias_variables[f"g{t2.id}"] == 1,
                    alias_variables[f"and{t1.id}_{t2.id}"],
                    z3.Not(
                        z3.Or(alias_variables[f"contains{t1.id}_{t2.id}"], alias_variables[f"contains{t2.id}_{t1.id}"])
                    )
                ) for t2 in transitions if t1.id != t2.id
            ])
            s.add(z3.Implies(v == 1, z3.Not(inner_or)))

            # It must hold that the priority of each member in the group is the same.
            inner_or = z3.Or([
                z3.And(
                    alias_variables[f"g{t2.id}"] == 1,
                    alias_variables[f"p{t2.id}"] != alias_variables[f"p{t1.id}"]
                ) for t2 in transitions if t1.id != t2.id
            ])
            s.add(z3.Implies(v == 1, z3.Not(inner_or)))

        # Maximize the size of the group.
        s.maximize(sum(alias_variables[f"g{t.id}"] for t in transitions))

        # Get the model solution and extract the selected transitions.
        result, model = s.check(), s.model()
        if result.r == z3.Z3_L_UNDEF:
            print(result, model)
            raise Exception("Unknown result.")
        if result.r == z3.Z3_L_FALSE:
            print(result, model)
            raise Exception("Unsatisfiable result.")

        # Find the target transitions that are part of the deterministic group.
        grouped_transitions: List[Transition] = [
            t for t in transitions if model.evaluate(alias_variables[f"g{t.id}"], model_completion=True) == 1
        ]
        deterministic_transition_groups.append(grouped_transitions)

        # Restore the state of the solver.
        s.pop()

        # Remove the transitions from the to be processed list.
        for t in grouped_transitions:
            transitions.remove(t)

    # Return the gathered groups.
    return deterministic_transition_groups


def process_non_deterministic_groups(
        deterministic_transition_groups: List[List[Union[DecisionNode, Transition]]],
        alias_variables: Dict[str, z3.ArithRef],
        and_table: Dict[int, Dict[int, bool]],
        contains_table: Dict[int, Dict[int, bool]]
) -> List[Union[DecisionNode, Transition]]:
    """Process the list of transition groups and convert them to decision nodes."""
    # Process the selected groups.
    non_deterministic_choices: List[Union[DecisionNode, Transition]] = []
    for g in deterministic_transition_groups:
        # Find remaining non-deterministic behavior within the deterministic group.
        deterministic_node = construct_deterministic_node(g, alias_variables, and_table, contains_table)

        # Add a deterministic decision node to the list of non-deterministic choices.
        non_deterministic_choices.append(deterministic_node)

    # Return the processed choices.
    return non_deterministic_choices


def construct_non_deterministic_node(
        transitions: List[Transition],
        alias_variables: Dict[str, z3.ArithRef],
        and_table: Dict[int, Dict[int, bool]],
        contains_table: Dict[int, Dict[int, bool]]
) -> Union[DecisionNode, Transition]:
    """Convert the list of transitions to a non-deterministic decision node."""
    # Preprocess transitions that overlap with all other available transitions.
    conflicting_transitions = remove_conflicting_transitions(and_table, transitions)

    # Assign the given transitions to groups that internally display deterministic behavior.
    deterministic_transition_groups: List[List[Union[DecisionNode, Transition]]] = get_non_deterministic_node_groups(
        transitions, alias_variables
    )

    # Process the selected groups.
    non_deterministic_choices: List[Union[DecisionNode, Transition]] = process_non_deterministic_groups(
        deterministic_transition_groups, alias_variables, and_table, contains_table
    )

    # Sort the non-deterministic choices and create a non-deterministic node.
    non_deterministic_choices += conflicting_transitions
    non_deterministic_choices.sort(key=lambda x: (x.priority, x.id))
    non_deterministic_node = DecisionNode(False, non_deterministic_choices, [])
    # TODO: lock ordering optimization.
    return non_deterministic_node


def select_non_deterministic_node(transitions: List[Transition]) -> DecisionNode:
    # Save the current state of the solver.
    s.push()

    # Create all common variables needed within the SMT solution and get the associated alias variables.
    alias_variables = create_smt_support_variables(
        transitions, s, include_and=True, include_equal=False, include_contains=True
    )

    # Calculate the overlap table.
    and_table: Dict[int, Dict[int, bool]] = dict()
    contains_table: Dict[int, Dict[int, bool]] = dict()
    result, model = s.check(), s.model()
    transitions.sort(key=lambda x: x.id)
    for t1 in transitions:
        and_table[t1.id] = {
            t2.id: model.evaluate(
                alias_variables[f"and{t1.id}_{t2.id}"], model_completion=True
            ) for t2 in transitions
        }
        contains_table[t1.id] = {
            t2.id: model.evaluate(
                alias_variables[f"contains{t1.id}_{t2.id}"], model_completion=True
            ) for t2 in transitions
        }

    # Ensure that the part of the group variable of each transition can only be zero or one.
    for t in transitions:
        v = alias_variables[f"g{t.id}"]
        s.add(z3.And(v >= 0, v < 2))

    # Select the non-deterministic choices made within the non-deterministic group.
    non_deterministic_node = construct_non_deterministic_node(
        transitions, alias_variables, and_table, contains_table
    )

    # Restore the state of the solver.
    s.pop()

    return non_deterministic_node
