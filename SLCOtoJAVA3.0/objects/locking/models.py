from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, List

import networkx as nx

# Avoid circular imports due to type checking.
if TYPE_CHECKING:
    from objects.ast.interfaces import SlcoStatementNode


class LockingNodeType(Enum):
    """
    The type of a given locking node.
    """
    # Entry nodes can perform lock actions.
    ENTRY = 1
    # Success and failure nodes can perform unlock actions.
    SUCCESS = 2
    FAILURE = 3
    # Passive nodes cannot perform lock or unlock actions.
    PASSIVE = 4


class LockingNode:
    """
    An object representing a locking node.

    The purpose of the class is to detach the locking system from the decision structure and control flow.
    """
    def __init__(self, partner, node_type: LockingNodeType):
        # The locks that need to be active to attain atomicity of itself and the successor nodes.
        self.required_locks = set()

        # The locks to be acquired and released in this node.
        self.locks_to_acquire = set()
        self.locks_to_release = set()

        # The object that the locking node is partnered with.
        self.partner = partner

        # The type of the node.
        self.node_type = node_type

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    def __hash__(self) -> int:
        return super().__hash__()


class AtomicNode:
    """
    An object representing a collection of locks that will make the given object behave as an atomic operation.
    """
    def __init__(self, partner: SlcoStatementNode):
        # The object that the atomic node is partnered with.
        self.partner = partner

        # The internal graph representation of the locking structure.
        self.graph = nx.DiGraph()

        # An atomic node has one entry point.
        self.entry_node: LockingNode = LockingNode(partner, LockingNodeType.ENTRY)

        # On top of that, an atomic node always has one success exit point and a failure exit point.
        self.success_exit: LockingNode = LockingNode(partner, LockingNodeType.SUCCESS)
        self.failure_exit: LockingNode = LockingNode(partner, LockingNodeType.FAILURE)

        # Add the default node types to the graph.
        self.graph.add_node(self.entry_node)
        self.graph.add_node(self.success_exit)
        self.graph.add_node(self.failure_exit)

        # Potential child atomic nodes of the atomic node.
        self.child_atomic_nodes: List[AtomicNode] = []

    def make_indifferent(self):
        """
        Merge the success and failure exits such that only one exit remains.
        """
        # Add all the edges from the failure exit to the success exit and remove the original.
        for n in self.graph.predecessors(self.failure_exit):
            self.graph.add_edge(n, self.success_exit)
        for n in self.graph.successors(self.failure_exit):
            self.graph.add_edge(self.success_exit, n)
        self.graph.remove_node(self.failure_exit)
        self.failure_exit = self.success_exit

    def include_atomic_node(self, node: AtomicNode):
        """
        Add the given atomic node to the graph of this atomic node.
        """
        # Add all of the nodes and edges from the target graph to the local graph.
        self.graph.update(node.graph)
        self.child_atomic_nodes.append(node)

    def mark_as_passive_recursively(self):
        """Mark all of the locking nodes within the atomic block as passive nodes."""
        for n in self.graph:
            n.node_type = LockingNodeType.PASSIVE

    def mark_as_passive(self):
        """Mark all of the entry and exit points within the atomic block as passive nodes."""
        for n in [self.entry_node, self.success_exit, self.failure_exit]:
            n.node_type = LockingNodeType.PASSIVE

