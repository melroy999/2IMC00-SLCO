from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, List, Set, Union, Tuple, Dict, Optional

import networkx as nx

# Avoid circular imports due to type checking.
import objects.ast.util as util
from objects.ast.models import Expression
from preprocessing.ast.simplification import simplify

if TYPE_CHECKING:
    from objects.ast.interfaces import SlcoStatementNode
    from objects.ast.models import VariableRef, Primary


class Lock:
    """
    An object representing a lock in the locking graph.
    """
    def __init__(self, ref: VariableRef, original_locking_node: LockingNode):
        # The variable reference that the lock request is associated with.
        self._original_ref: VariableRef = ref

        # Rewrite rules that have been applied to the request during movement operations.
        self.rewrite_rules_list: List[Tuple[VariableRef, Union[Expression, Primary]]] = []

        # The reference rewritten by the rewrite rules. Default value is the original reference itself.
        self.ref: VariableRef = ref

        # A flag indicating whether the lock request is location sensitive or not. I.e., depends on the control flow.
        self.is_location_sensitive: bool = False

        # A flag indicating that is set to true when the index cannot be locked with the locks currently available.
        self.is_dirty: bool = False

        # The locking node that this request was created in.
        self.original_locking_node: LockingNode = original_locking_node

        # The atomic node that the lock request originated from.
        self.parent: AtomicNode = original_locking_node.parent

    def prepend_rewrite_rule(self, rule: Tuple[VariableRef, Union[Expression, Primary]]):
        """
        Add the given rewrite rule to the start of the rewrite rule list and update the master dictionary.
        """
        # Rewriting is only used for array type variables.
        if self._original_ref.var.is_array:
            # Add the rule to the start of the list.
            self.rewrite_rules_list.insert(0, rule)

            # Generate the dictionary.
            rewrite_rules: Dict[VariableRef, Union[Expression, Primary]] = dict()
            for variable, replacement in self.rewrite_rules_list:
                if variable.var.is_array:
                    # Rewrite target_variable appropriately if it has an index.
                    target_variable = util.copy_node(variable, dict(), dict())
                    target_variable.index = util.copy_node(target_variable.index, dict(), rewrite_rules)
                else:
                    target_variable = variable

                # Apply the rewrite rule to the replacement statement and note down the change of value.
                rewrite_rules[target_variable] = util.copy_node(replacement, dict(), rewrite_rules)

            # Apply the rewrite rules to the index and create a new object.
            # Simplify the index to remove superfluous brackets.
            self.ref = util.copy_node(self._original_ref, dict(), dict())
            self.ref.index = simplify(util.copy_node(self.ref.index, dict(), rewrite_rules))

    def __hash__(self):
        # Use the hash function of the targeted variable reference.
        return hash(self._original_ref)

    def __eq__(self, o: object) -> bool:
        # Besides the references variable, the parent needs to be equivalent too due to different graph localities.
        if isinstance(o, Lock):
            return self.parent == o.parent and self._original_ref == o._original_ref
        return False

    def __le__(self, o: object):
        """
        Compare lock requests with a shorter <= syntax. Returns true if the lock should be at the same level or above
        the lock that is being compared to.
        """
        if isinstance(o, Lock):
            # The type of comparison made differs based on whether lock identities have been assigned or not.
            if self.ref.var.lock_id != -1 and self.ref.var.lock_id < o.ref.var.lock_id:
                # If lock identities are set, compare the identities. If the identity is smaller, return true.
                # Equality is not checked, since the same rules will apply as for comparison without lock identities.
                return True
            if self.ref.var.lock_id != -1 and self.ref.var.lock_id > o.ref.var.lock_id:
                return False
            elif self.ref.var != o.ref.var:
                # Otherwise, locks can only be compared if the variables match. Hence, return false.
                return False
            elif not self.ref.var.is_array:
                # There is no index to be compared, so the references are equivalent.
                return True
            else:
                # If either of the two locks has a non-constant index, then the operator will always yield true.
                if isinstance(self.ref.index, Expression) or self.ref.index.value is None:
                    return True
                elif isinstance(o.ref.index, Expression) or o.ref.index.value is None:
                    return True
                else:
                    # Compare the constant values of the two objects.
                    return self.ref.index.value <= o.ref.index.value
        return False

    def __repr__(self):
        if self.ref == self._original_ref:
            return str(self.ref)
        else:
            return "%s(%s)" % (self.ref, self._original_ref)


class LockingNodeType(Enum):
    """
    The type of a given locking node.
    """
    ENTRY = 1
    SUCCESS = 2
    FAILURE = 3


class LockingNode:
    """
    An object representing a locking node.

    The purpose of the class is to detach the locking system from the decision structure and control flow.
    """
    def __init__(self, partner, node_type: LockingNodeType, parent: AtomicNode):
        # The locks that are requested and released by this locking node.
        self.locks_to_acquire: Set[Lock] = set()
        self.locks_to_release: Set[Lock] = set()

        # The object that the locking node is partnered with.
        self.partner = partner

        # The type of the node.
        self.node_type = node_type

        # The atomic node that created this locking node.
        self.parent = parent

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
        self.entry_node: LockingNode = LockingNode(partner, LockingNodeType.ENTRY, self)

        # On top of that, an atomic node always has one success exit point and a failure exit point.
        self.success_exit: LockingNode = LockingNode(partner, LockingNodeType.SUCCESS, self)
        self.failure_exit: LockingNode = LockingNode(partner, LockingNodeType.FAILURE, self)

        # Add the default node types to the graph.
        self.graph.add_node(self.entry_node)
        self.graph.add_node(self.success_exit)
        self.graph.add_node(self.failure_exit)

        # Potential child atomic nodes of the atomic node.
        self.child_atomic_nodes: List[AtomicNode] = []

        # A pointer to the node's parent, if it exists.
        self.parent: Optional[AtomicNode] = None

    def include_atomic_node(self, node: AtomicNode):
        """
        Add the given atomic node to the graph of this atomic node.
        """
        # Add all of the nodes and edges from the target graph to the local graph.
        self.graph.update(node.graph)
        self.child_atomic_nodes.append(node)
        node.parent = self

    def mark_indifferent(self):
        """
        Turn the atomic node into a node that is indifferent to the evaluation of the partnered statement.

        To achieve this, it is assumed that the statement will always be successful.
        """
        logging.debug(f" - Marking the atomic node \"{self.partner}\" as indifferent")
        if self.success_exit == self.failure_exit:
            raise Exception("The exits are already indifferent!")

        # Copy all of the failure exit's connections into the success exit.
        for n in self.graph.predecessors(self.failure_exit):
            self.graph.add_edge(n, self.success_exit)
        for n in self.graph.successors(self.failure_exit):
            self.graph.add_edge(self.success_exit, n)
        self.graph.remove_node(self.failure_exit)
        self.failure_exit = self.success_exit
