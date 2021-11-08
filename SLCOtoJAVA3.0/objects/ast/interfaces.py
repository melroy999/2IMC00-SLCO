from __future__ import annotations

from abc import ABCMeta
from collections.abc import Iterable
from typing import Optional, Set, Dict, List

import networkx as nx

import objects.ast.models as models
from rendering.util.to_smt import to_smt, is_true, is_false, is_equivalent, is_negation_equivalent


# INTERFACES/META-CLASSES
class SlcoNode(metaclass=ABCMeta):
    """
    A metaclass representing a node in the SLCO AST.
    """
    parent = None


class SlcoStructuralNode(SlcoNode, Iterable, metaclass=ABCMeta):
    """
    A metaclass for a node within the SLCO AST that has one or more child node fields.
    """


class SlcoEvaluableNode(SlcoNode, metaclass=ABCMeta):
    """
    A metaclass that provides helper functions for evaluating SMT compatible types.
    """
    def is_true(self) -> bool:
        """Evaluate whether the statement always holds true. May throw an exception if not a boolean smt statement."""
        return is_true(self)

    def is_false(self) -> bool:
        """Evaluate whether the statement never holds true. May throw an exception if not a boolean smt statement."""
        return is_false(self)

    def is_equivalent(self, target: SlcoEvaluableNode) -> bool:
        """Evaluate whether the given statements have the same solution space."""
        return is_equivalent(self, target)

    def is_negation_equivalent(self, target: SlcoEvaluableNode) -> bool:
        """Evaluate whether this statement and the negation of the given statement have the same solution space."""
        return is_negation_equivalent(self, target)

    @property
    def smt(self):
        """Get the smt statement associated to the object."""
        return to_smt(self)


class SlcoVerifiableNode:
    """
    A metaclass that provides helper functions for formal verification purposes.
    """
    # The vercors statements associated to the node.
    vercors_statements: Set[str] = None


class SlcoVersioningNode(metaclass=ABCMeta):
    """
    A metaclass that helps tracking changes made to objects for verification purposes.
    """
    # Version control.
    exclude_statement: bool = False
    produced_statement: bool = False
    original_statement: Optional[SlcoVersioningNode] = None

    def get_original_statement(self):
        """Get the original version of the statement."""
        if self.original_statement is None:
            return self
        else:
            return self.original_statement.get_original_statement()


class SlcoLockableNode(metaclass=ABCMeta):
    """
    A metaclass that provides helper functions for locking objects.
    """
    # Before statement:
    # Phase 1: Initial locks to request (Possibly multiple phases), including conflict resolutions, minus the violators.
    locks_to_acquire: Set[models.LockRequest] = set()
    locks_to_acquire_phases: List[Set[models.LockRequest]] = []

    # Phase 2: Acquire locks that were unpacked due to conflicts but will be acquired successfully in the previous step.
    unpacked_lock_requests: Set[models.LockRequest] = set()

    # Phase 3: Release the locks added by the conflict resolution that are not part of the original lock requests.
    conflict_resolution_lock_requests: Set[models.LockRequest] = set()

    # After statement:
    # Phase 4: Release the locks that are no longer required after the execution of the statement.
    locks_to_release: Set[models.LockRequest] = set()


class SlcoStatementNode(SlcoStructuralNode, SlcoVersioningNode, SlcoLockableNode, metaclass=ABCMeta):
    """
    A metaclass that provides helper functions and variables for statement-level objects in the SLCO framework.
    """
    # Avoid recalculating structural data.
    variable_references: Set[models.VariableRef] = None
    class_variable_references: Set[models.VariableRef] = None
    variable_dependency_graph: nx.DiGraph = None
    class_variable_dependency_graph: nx.DiGraph = None
