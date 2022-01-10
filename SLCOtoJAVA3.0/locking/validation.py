from __future__ import annotations

import logging
from typing import Set, TYPE_CHECKING
from objects.ast.models import Assignment, DecisionNode

import networkx as nx

from objects.ast.util import get_class_variable_references
from objects.locking.models import Lock

if TYPE_CHECKING:
    from objects.ast.models import VariableRef
    from objects.locking.models import AtomicNode, LockingNode, LockRequest


def validate_locking_structure_integrity(model: AtomicNode):
    """
    Verify that the locking structure does not violate the semantics.
        - All locks used in base-level lockable components need to have been acquired before use.
        - On each control flow path, all locks that are acquired are eventually released.
        - All locks that are released have been acquired earlier on in the path.
        - All locks that are acquired have not already been acquired by a previous locking node.
    """
    # Find all starting points in the graph.
    starting_points = [n for n in model.graph.nodes if len(list(model.graph.predecessors(n))) == 0]

    # Raise an exception of any of the paths does not get through the validation.
    if not all(validate_locking_node_integrity(n, model.graph, set()) for n in starting_points):
        raise Exception(f"The atomic node of \"{model.partner}\" has a path that violates the locking semantics.")
    if not all(validate_locking_instruction_integrity(n, model.graph, set()) for n in starting_points):
        raise Exception(f"The atomic node of \"{model.partner}\" has a path that violates the locking semantics.")


def validate_locking_node_integrity(
        n: LockingNode, graph: nx.DiGraph, acquired_locks: Set[Lock]
) -> bool:
    """
    Verify the following:
        - All locks used in base-level lockable components need to have been acquired before use.
    """
    # The entry node of the associated atomic node.
    entry_node = n.parent.entry_node

    # Add the (original) variable references requested by the current locking node.
    acquired_locks.update(n.locks_to_acquire)

    # All locks used in base-level lockable components need to have been acquired before use.
    target_variables: Set[VariableRef] = set()
    if len(n.partner.locking_atomic_node.child_atomic_nodes) == 0:
        target_variables = get_class_variable_references(n.partner)
    elif isinstance(n.partner, Assignment) and len(n.partner.locking_atomic_node.child_atomic_nodes) == 1:
        target_variables = get_class_variable_references(n.partner.left)

    # Ensure that all variables used by the statement are locked.
    target_locks: Set[Lock] = {Lock(r, entry_node) for r in target_variables}
    if len(target_locks) > 0 and len(target_locks.difference(acquired_locks)) > 0:
        logging.info("Locking structure validation failed. The statement has variables that are not locked.")
        return False

    # The path so far is valid. Check if successors are valid too.
    return all(
        validate_locking_node_integrity(n, graph, set(acquired_locks)) for n in graph.successors(n)
    )


def validate_locking_instruction_integrity(
        n: LockingNode, graph: nx.DiGraph, acquired_lock_requests: Set[LockRequest]
) -> bool:
    """
    Verify the following:
        1. Any locks added in this node should not yet be present in the acquired lock requests list.
        2. Lock requests that are released need to be present in the acquired lock requests list.
        3. If the node has no successors, the list of acquired locks needs to be equivalent to the locks to be released.
        ~4. Decision nodes should not have locks to release or request.
    """
    instructions = n.locking_instructions

    # ~4. Decision nodes should not have locks to release or request.
    if isinstance(n.partner, DecisionNode) and instructions.has_locks():
        logging.info("Locking structure validation failed. The decision nodes have locks to release or request.")
        return False

    # 1. Any locks added in this node should not yet be present in the acquired lock requests list.
    if len(acquired_lock_requests.intersection(instructions.locks_to_acquire)) > 0 \
            or len(acquired_lock_requests.intersection(instructions.unpacked_lock_requests)):
        logging.info("Locking structure validation failed. Locks acquired by the node are already present beforehand.")
        return False

    # Add the locks acquired by the current node to the list.
    acquired_lock_requests.update(instructions.locks_to_acquire)
    acquired_lock_requests.update(instructions.unpacked_lock_requests)

    # 2. Lock requests that are released need to be present in the acquired lock requests list.
    if len(instructions.locks_to_release.difference(acquired_lock_requests)) != 0:
        logging.info("Locking structure validation failed. Locks released by the node weren't present beforehand.")
        return False

    # Release locks that are released by the current node.
    acquired_lock_requests.difference_update(instructions.locks_to_release)

    if len(list(graph.successors(n))) == 0:
        # 3. No locks are allowed to remain after the lock release is done.
        if len(acquired_lock_requests) > 0:
            logging.info("Locking structure validation failed. Locks remain at the end of the control flow.")
            return False

    # The path so far is valid. Check if successors are valid too.
    return all(
        validate_locking_instruction_integrity(n, graph, set(acquired_lock_requests)) for n in graph.successors(n)
    )
