from __future__ import annotations

from abc import ABCMeta
from collections.abc import Iterable
from typing import Optional

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


class SlcoStatementNode(SlcoNode, metaclass=ABCMeta):
    """
    A metaclass that provides helper functions and variables for statement-level objects in the SLCO framework.
    """
    exclude_statement: bool = False
    produced_statement: bool = False
    original_statement: Optional[SlcoStatementNode] = None

    def get_original_statement(self):
        """Get the original version of the statement."""
        if self.original_statement is None:
            return self
        else:
            return self.original_statement.get_original_statement()
