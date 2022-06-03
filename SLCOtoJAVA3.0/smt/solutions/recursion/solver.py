from typing import List, Dict, Union, Set

import numpy as np
from z3 import z3

from objects.ast.models import Transition, DecisionNode
from smt.solutions.solver import DecisionStructureSolver


class RecursionDecisionStructureSolver(DecisionStructureSolver):
    """A class that converts a list of transitions to a decision structure, using recursion for nested structures."""

    # noinspection PyMethodMayBeStatic
    def get_encompassing_guard_statement(self, transitions: List[Transition]) -> Transition:
        """Get the transition that contains the encompassing guard statement of the nested group."""
        # By default, return the transition with the lowest id.
        transitions.sort(key=lambda t: t.id)
        return transitions[0]

    def create_deterministic_node(
            self, transitions: List[Transition], alias_variables: Dict[str, z3.ArithRef]
    ) -> Union[DecisionNode, Transition]:
        """Select non-deterministic groups within the given list of transitions."""
        # Overlap may still exist within the deterministic group. Group by overlap.
        index_to_transition: Dict[int, Transition] = {i: t for i, t in enumerate(transitions)}

        # Use matrix multiplications to find the overlapping groups. Multiply by the length to ensure max propagation.
        truth_matrix = np.matrix(
            [[self.and_truth_table[t1.id][t2.id] for t2 in transitions] for t1 in transitions], dtype=bool
        )
        propagated_truth_matrix = np.linalg.matrix_power(truth_matrix, len(transitions))

        # Extract the groups.
        decision_groups: List[List[Transition]] = []
        grouped_indices: Set[int] = set()
        for i, t1 in enumerate(transitions):
            # Pass if already processed.
            if i in grouped_indices:
                continue

            # Find all transitions that overlap.
            target_indices: List[int] = [j for j, t2 in enumerate(transitions) if propagated_truth_matrix[i, j]]
            decision_groups.append([index_to_transition[j] for j in target_indices])
            grouped_indices.update(target_indices)

        # All decision groups will become nested non-deterministic nodes.
        deterministic_choices: List[Union[DecisionNode, Transition]] = []
        for g in decision_groups:
            if len(g) > 1:
                # Find the encompassing statement, which may act as the decision node's guard statement.
                guard_statement = self.get_encompassing_guard_statement(g)

                # Create a non-deterministic block.
                decision_node = self.create_non_deterministic_node(g, alias_variables)
                decision_node.guard_statement = guard_statement
                deterministic_choices.append(decision_node)
            else:
                deterministic_choices.append(g[0])

        # Sort the transitions and create a deterministic node.
        deterministic_choices.sort(key=lambda x: (x.priority, x.id))
        deterministic_node = DecisionNode(True, deterministic_choices, [])
        return deterministic_node
