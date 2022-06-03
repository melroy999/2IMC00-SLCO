from typing import List, Dict, Union

from z3 import z3

from objects.ast.models import Transition, DecisionNode
from smt.solutions.solver import DecisionStructureSolver


class OptimalDecisionStructureSolver(DecisionStructureSolver):
    """A class that converts a list of transitions to a decision structure, solved using SMT only."""

    @staticmethod
    def get_group_variable_bounds(v: z3.ArithRef, transitions: List[Transition]) -> z3.ArithRef:
        """Get the bounds to use for the group variable within the smt model."""
        # Make sure that there are enough groups available to assign every transition to its own group.
        return z3.And(v >= 0, v < len(transitions))

    def create_non_deterministic_node_smt_model_constraints(
            self, transitions: List[Transition], alias_variables: Dict[str, z3.ArithRef]
    ) -> None:
        """Add the constraints needed to extract groups within non-deterministic nodes."""
        # Create the rules needed to construct valid groupings.
        for t1 in transitions:
            v = alias_variables[f"g{t1.id}"]

            # Provide enough constraints to ensure that each transition can end up in its own group.
            for k in range(len(transitions)):
                # It must hold that the transition has no overlap with any of the members in the same group.
                self.create_non_deterministic_node_smt_model_overlap_constraints(
                    t1, v, transitions, alias_variables, k
                )

                # It must hold that the priority of each member in the group is the same.
                self.create_non_deterministic_node_smt_model_priority_constraints(
                    t1, v, transitions, alias_variables, k
                )

        # Maximize the size of the group.
        self.create_non_deterministic_node_smt_model_optimization_constraint(transitions, alias_variables)

    def create_non_deterministic_node_smt_model_optimization_constraint(
            self, transitions: List[Transition], alias_variables: Dict[str, z3.ArithRef]
    ) -> None:
        """Add the maximization constraints to the model."""
        # Minimize, since we want the lowest number of groups.
        self.solver.minimize(sum(alias_variables[f"g{t.id}"] for t in transitions))

    def get_non_deterministic_node_groups(
            self, transitions: List[Transition], alias_variables: Dict[str, z3.ArithRef]
    ) -> List[List[Union[DecisionNode, Transition]]]:
        """Convert the list of transitions to groups to be contained within a non-deterministic decision node."""
        # Return an empty list if no transitions are present.
        if len(transitions) == 0:
            return []

        # Assign the given transitions to groups that internally display deterministic behavior.
        deterministic_transition_groups: List[List[Union[DecisionNode, Transition]]] = []

        # Save the current state of the solver.
        self.solver.push()

        # Introduce the appropriate model constraints for the group selection.
        self.create_non_deterministic_node_smt_model_constraints(transitions, alias_variables)

        # Get the model solution.
        result, model = self.solve_smt_model()

        # Find the groupings. Elements within the same group assigned should be added to the same group.
        for k, _ in enumerate(transitions):
            grouped_transitions: List[Transition] = []
            for t in transitions:
                if model.evaluate(alias_variables[f"g{t.id}"], model_completion=True) == k:
                    grouped_transitions.append(t)

            # Skip if no transitions.
            if len(grouped_transitions) == 0:
                continue

            # Add the group to the list of deterministic transitions.
            deterministic_transition_groups.append(grouped_transitions)

        # Restore the state of the solver.
        self.solver.pop()

        # Return the gathered groups.
        return deterministic_transition_groups
