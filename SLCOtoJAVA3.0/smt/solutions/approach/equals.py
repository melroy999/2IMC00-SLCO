from typing import Dict, List

from z3 import z3

from objects.ast.models import Transition
from smt.solutions.solver import DecisionStructureSolver


class EqualsDecisionStructureSolver(DecisionStructureSolver):
    """A class that converts a list of transitions to a decision structure, using equality for more refined groups."""

    def create_non_deterministic_node_smt_model_overlap_constraints(
            self,
            t1: Transition,
            v: z3.ArithRef,
            transitions: List[Transition],
            alias_variables: Dict[str, z3.ArithRef],
            target_group: int = 1
    ) -> None:
        """Add the overlap constraints to the model."""
        inner_or = z3.Or([
            z3.And(
                alias_variables[f"g{t2.id}"] == target_group,
                alias_variables[f"and{t1.id}_{t2.id}"],
                z3.Not(alias_variables[f"ieq{t1.id}_{t2.id}"])
            ) for t2 in transitions if t1.id != t2.id
        ])
        self.solver.add(z3.Implies(v == target_group, z3.Not(inner_or)))
