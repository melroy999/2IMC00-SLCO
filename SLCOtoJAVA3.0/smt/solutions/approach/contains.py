from typing import Dict, List

from z3 import z3

from objects.ast.models import Transition
from smt.solutions.solver import DecisionStructureSolver


class ContainsDecisionStructureSolver(DecisionStructureSolver):
    """A class that converts a list of transitions to a decision structure, using contains for more refined groups."""

    def get_encompassing_guard_statement(self, transitions: List[Transition]) -> Transition:
        """Get the transition that contains the encompassing guard statement of the nested group."""
        # Select the transition that contains the most transitions.
        number_of_contains = {
            t1: len([t2 for t2 in transitions if self.contains_truth_table[t1.id][t2.id]]) for t1 in transitions
        }
        transitions.sort(reverse=True, key=lambda t1: (number_of_contains[t1], -t1.id))
        return transitions[0]

    def create_non_deterministic_node_smt_model_overlap_constraints(
            self,
            t1: Transition,
            v: z3.ArithRef,
            transitions: List[Transition],
            alias_variables: Dict[str, z3.ArithRef],
            target_group: int
    ) -> None:
        """Add the overlap constraints to the model."""
        inner_or = z3.Or([
            z3.And(
                alias_variables[f"g{t2.id}"] == target_group,
                alias_variables[f"and{t1.id}_{t2.id}"],
                z3.Not(
                    z3.Or(alias_variables[f"contains{t1.id}_{t2.id}"], alias_variables[f"contains{t2.id}_{t1.id}"])
                )
            ) for t2 in transitions if t1.id != t2.id
        ])
        self.solver.add(z3.Implies(v == target_group, z3.Not(inner_or)))
