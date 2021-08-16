from __future__ import annotations

import networkx as nx
import objects.ast.models as models

from typing import Set
from abc import ABCMeta
from collections.abc import Iterable

from rendering.util.to_smt import to_smt, is_true, is_false, is_equivalent, is_negation_equivalent
from util.smt import z3_always_holds, z3_never_holds, z3_is_equivalent


# INTERFACES/META-CLASSES
class Preprocessable(Iterable, metaclass=ABCMeta):
    """
    A simple class extension that provides helper functions for recursive AST node pre-processing.
    """
    def preprocess(self) -> None:
        """Preprocess the structure of the AST to simplify and generalize."""
        for o in self:
            if isinstance(o, Preprocessable):
                o.preprocess()


class Evaluable(metaclass=ABCMeta):
    """
    A simple class extension that provides helper functions for evaluating SMT compatible types.
    """
    def is_true(self) -> bool:
        """Evaluate whether the statement always holds true. May throw an exception if not a boolean smt statement."""
        return is_true(self)

    def is_false(self) -> bool:
        """Evaluate whether the statement never holds true. May throw an exception if not a boolean smt statement."""
        return is_false(self)

    def is_equivalent(self, target: Evaluable) -> bool:
        """Evaluate whether the given statements have the same solution space."""
        return is_equivalent(self, target)

    def is_negation_equivalent(self, target: Evaluable) -> bool:
        """Evaluate whether this statement and the negation of the given statement have the same solution space."""
        return is_negation_equivalent(self, target)

    @property
    def smt(self):
        """Get the smt statement associated to the object."""
        return to_smt(self)


class DFS(Iterable, metaclass=ABCMeta):
    """
    A simple helper class that allows for a Depth-First-Search (DFS) through the AST, with the nodes being reported upon
    the return path.
    """
    def __iter_dfs__(self):
        """Iterate through all objects in the AST through a DFS, with nodes being reported upon the return path."""
        if isinstance(self, Iterable):
            for o in self:
                if isinstance(o, DFS):
                    yield from o.__iter_dfs__()
                else:
                    yield o
        yield self


class Lockable(DFS, metaclass=ABCMeta):
    """
    A simple class extension that provides helper functions for fetching lockable objects in an AST structure.
    """
    def get_variable_references(self) -> Set[models.VariableRef]:
        """
        Get a list of all the variables that have been referenced to by the statement. Note that variables used in
        composites have been adjusted through rewrite rules to compensate for assignments.
        """
        referenced_variables = set()
        for o in self.__iter_dfs__():
            if isinstance(o, models.VariableRef):
                referenced_variables.add(o)
        return referenced_variables

    def get_class_variable_references(self) -> Set[models.VariableRef]:
        """
        Get a list of all the class variables that have been referenced to by the statement. Note that variables used in
        composites have been adjusted through rewrite rules to compensate for assignments.
        """
        return set(v for v in self.get_variable_references() if v.var.is_class_variable)

    def get_variable_dependency_graph(self) -> nx.DiGraph:
        """Get a variable dependency graph for the variables within the statement."""
        graph = nx.DiGraph()
        references: Set[models.VariableRef] = set(self.get_variable_references())
        while len(references) > 0:
            target = references.pop()
            graph.add_node(target.var)
            if target.index is not None:
                sub_references: Set[models.VariableRef] = target.index.get_variable_references()
                for r in sub_references:
                    graph.add_edge(target.var, r.var)
        return graph

    def get_class_variable_dependency_graph(self) -> nx.DiGraph:
        """Get a variable dependency graph for the class variables within the statement."""
        graph = self.get_variable_dependency_graph()
        sm_variables = [n for n in graph.nodes if not n.is_class_variable]
        graph.remove_nodes_from(sm_variables)
        return graph


class Copyable(metaclass=ABCMeta):
    """
    A simple class extension that provides helper functions for creating copies of objects in an AST structure.
    """

    def create_copy(self, rewrite_rules: dict, is_first: bool = True) -> Copyable:
        """Create a copy of the expression with the given rewrite rules applied."""
        pass
