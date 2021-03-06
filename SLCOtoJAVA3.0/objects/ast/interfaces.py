from __future__ import annotations

from abc import ABCMeta
from collections.abc import Iterable
from typing import Optional, Set, TYPE_CHECKING

import networkx as nx

from objects.util import get_incremental_id
from smt.conversion import to_smt
from smt.solving import is_true, is_false, is_equivalent, is_negation_equivalent, exists_lte

# Avoid circular imports due to type checking.
if TYPE_CHECKING:
    from objects.ast.models import VariableRef
    from objects.locking.models import AtomicNode


# INTERFACES/META-CLASSES
class SlcoNode(metaclass=ABCMeta):
    """
    A metaclass representing a node in the SLCO AST.
    """
    def __init__(self) -> None:
        self.parent = None
        self.id = get_incremental_id()


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

    def exists_lte(self, target: SlcoEvaluableNode):
        """
        Evaluate whether there exists a solution for which this statement is less than or equal to the given statement.
        """
        return exists_lte(self, target)

    @property
    def smt(self):
        """Get the smt statement associated to the object."""
        return to_smt(self)


class SlcoVerifiableNode(SlcoNode):
    """
    A metaclass that provides helper functions for formal verification purposes.
    """
    # The vercors statements associated to the node.
    vercors_statements: Set[str] = None


class SlcoVersioningNode(SlcoNode, metaclass=ABCMeta):
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
    # The atomic node and locking graph associated with the object.
    locking_atomic_node: Optional[AtomicNode] = None


class SlcoStatementNode(SlcoStructuralNode, SlcoVersioningNode, SlcoLockableNode, metaclass=ABCMeta):
    """
    A metaclass that provides helper functions and variables for statement-level objects in the SLCO framework.
    """
    # Avoid recalculating structural data.
    variable_references: Set[VariableRef] = None
    class_variable_references: Set[VariableRef] = None
    variable_dependency_graph: nx.DiGraph = None
    class_variable_dependency_graph: nx.DiGraph = None
