from smt.solutions.approach.contains import ContainsDecisionStructureSolver
from smt.solutions.approach.equals import EqualsDecisionStructureSolver
from smt.solutions.optimal.solver import OptimalDecisionStructureSolver
from smt.solutions.recursion.solver import RecursionDecisionStructureSolver
from smt.solutions.solver import DecisionStructureSolver


class GreedyBaseDecisionStructureSolver(DecisionStructureSolver):
    """A greedy decision structure solver that meets the minimum requirements."""
    pass


class GreedyEqualsNestedDecisionStructureSolver(EqualsDecisionStructureSolver, RecursionDecisionStructureSolver):
    """
    A greedy and nested decision structure solver that merges equal statements into nested non-deterministic nodes.
    """
    pass


class GreedyContainsNestedDecisionStructureSolver(ContainsDecisionStructureSolver, RecursionDecisionStructureSolver):
    """
    A greedy and nested decision structure solver that merges transitions into nested non-deterministic nodes based
    on the contains relations between said transitions.
    """
    pass


class OptimalBaseDecisionStructureSolver(GreedyBaseDecisionStructureSolver, OptimalDecisionStructureSolver):
    """An optimal decision structure solver that meets the minimum requirements."""
    pass


class OptimalEqualsNestedDecisionStructureSolver(
    GreedyEqualsNestedDecisionStructureSolver, OptimalDecisionStructureSolver
):
    """
    An optimal and nested decision structure solver that merges equal statements into nested non-deterministic nodes.
    """
    pass


class OptimalContainsNestedDecisionStructureSolver(
    GreedyContainsNestedDecisionStructureSolver, OptimalDecisionStructureSolver
):
    """
    An optimal and nested decision structure solver that merges transitions into nested non-deterministic nodes based
    on the contains relations between said transitions.
    """
    pass
