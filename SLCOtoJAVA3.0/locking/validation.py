from __future__ import annotations

from typing import Set, TYPE_CHECKING
from objects.ast.models import Assignment

import networkx as nx

from objects.ast.util import get_class_variable_references

if TYPE_CHECKING:
    from objects.ast.models import VariableRef
    from objects.locking.models import AtomicNode, LockingNode


def validate_path_integrity(model: AtomicNode):
    """
    Verify the following:
        - On each control flow path, all locks that are acquired are eventually released.
        - All locks that are released have been acquired earlier on in the path.
        - All locks that are acquired have not been requested previously.
        - All locks that are required for the target operation have been acquired beforehand.
    """
    # Find all starting points in the graph.
    starting_points = [n for n in model.graph.nodes if len(list(model.graph.predecessors(n))) == 0]

    # Raise an exception of any of the paths does not get through the validation.
    if not all(check_path_integrity(n, model.graph, set()) for n in starting_points):
        raise Exception(f"The atomic node of \"{model.partner}\" has a path that violates the validation.")


def check_path_integrity(n: LockingNode, graph: nx.DiGraph, acquired_locks: Set[VariableRef]) -> bool:
    """
    Verify the following:
        1. If the node has no successors, the list of acquired locks needs to be equivalent to the locks to be released.
        2. Any locks added in this node should not yet be present in the acquired locks list.
        3. Locks used at the base level of the partner statement need to be present in the acquired locks list.
        4. Locks that are released need to be present in the acquired locks list.
        5. The locks to be released and locks to be acquired sets do not have lock requests in common.
    """
    # 2. Any locks added in this node should not yet be present in the acquired locks list.
    if len(acquired_locks.intersection(n.locks_to_acquire)) > 0:
        return False

    # 5. The locks to be released and locks to be acquired sets do not have lock requests in common.
    if len(n.locks_to_acquire.intersection(n.locks_to_release)) > 0:
        return False

    # Add the locks acquired by the current node to the list.
    acquired_locks.update(n.locks_to_acquire)

    # 3. Locks used by partner statement in base-level nodes need to be present in the acquired locks list.
    # TODO: isn't n.is_base_level kind of equivalent to len(n.partner.locking_atomic_node.child_atomic_nodes) == 0 for
    #  all but assignments?
    target_variables = None
    if len(n.partner.locking_atomic_node.child_atomic_nodes) == 0:
        target_variables = get_class_variable_references(n.partner)
    elif isinstance(n.partner, Assignment) and len(n.partner.locking_atomic_node.child_atomic_nodes) == 1:
        target_variables = get_class_variable_references(n.partner.left)

    if target_variables is not None:
        # Ensure that all variables used by the statement are locked.
        if len(target_variables) > 0 and len(target_variables.difference(acquired_locks)) > 0:
            return False

    # 4. Locks that are released need to be present in the acquired locks list.
    if len(n.locks_to_release.difference(acquired_locks)) != 0:
        return False

    # Release locks that are released by the current node.
    acquired_locks.difference_update(n.locks_to_release)

    if len(list(graph.successors(n))) == 0:
        # 1. No locks are allowed to remain after the lock release is done.
        if len(acquired_locks) > 0:
            return False

    # The model is valid. Check if successors are valid too.
    return all(check_path_integrity(n, graph, set(acquired_locks)) for n in graph.successors(n))
