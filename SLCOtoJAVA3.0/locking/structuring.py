from typing import Dict, Set

import networkx as nx

from locking.validation import validate_path_integrity
from objects.ast.interfaces import SlcoLockableNode
from objects.ast.models import Primary, Composite, Transition, Assignment, Expression, StateMachine, Class
from objects.ast.util import get_class_variable_references
from objects.locking.models import AtomicNode, LockingNode, Lock
from objects.locking.visualization import render_locking_structure


def get_locking_structure(model: Transition):
    """
    Construct a locking structure for the given decision structure.
    """
    # Assert that no locking identities have been assigned yet.
    state_machine: StateMachine = model.parent
    _class: Class = state_machine.parent
    assert(all(v.lock_id == -1 for v in _class.variables))

    # Construct the locking structure for the object.
    create_locking_structure(model)

    # Generate the base-level locking data and add it to the locking structure.
    for s in model.statements:
        generate_base_level_locking_data(s.locking_atomic_node)

    # Perform a movement step prior to knowing the lock identities.
    # Two passes are required: one to get the non-constant array variables to the right position in the graph, and a
    # second to compensate for the effect that the movement of the aforementioned non-constant array indices may have
    # on the location of constant valued indices.
    for s in model.statements:
        restructure_lock_acquisitions(s.locking_atomic_node)

    # Render the locking structure.
    for s in model.statements:
        render_locking_structure(s.locking_atomic_node)


def finalize_locking_structure(model: Transition):
    """
    Finalize the locking structure for the given decision structure based on the given lock priorities.
    """
    # Assert that the locking identities have been assigned.
    state_machine: StateMachine = model.parent
    _class: Class = state_machine.parent
    assert(any(v.lock_id != -1 for v in _class.variables))

    # Perform another restructuring pass.
    for s in model.statements:
        restructure_lock_acquisitions(s.locking_atomic_node)

    # Move the lock releases to the appropriate location.
    for s in model.statements:
        restructure_lock_releases(s.locking_atomic_node)

    # Render the locking structure.
    for s in model.statements:
        render_locking_structure(s.locking_atomic_node)


def is_boolean_statement(model) -> bool:
    """
    Determine whether the given statement yields a boolean result or not.
    """
    if isinstance(model, Primary):
        if model.sign == "not" and model.value is None:
            # Values with a negation that are non-constant are boolean statements.
            return True
        elif model.ref is not None and model.ref.var.is_boolean:
            # Primaries referencing to boolean variables are boolean statements.
            return True
        elif model.body is not None:
            # Determine if the nested object is a boolean statement or not.
            return is_boolean_statement(model.body)
    elif isinstance(model, Expression) and model.op not in ["+", "-", "*", "/", "%", "**"]:
        # Expressions not using mathematical operators are boolean statements.
        return True
    return False


def create_locking_structure(model) -> AtomicNode:
    """
    Create a DAG of locking nodes that will dictate which locks will need to be requested/released at what position.

    The returned node is the encompassing atomic node of the model.
    """
    if isinstance(model, Transition):
        # The guard expression of the transition needs to be included in the main structure. Other statements do not.
        for s in model.statements:
            create_locking_structure(s)

        # Chain the guard statement with the overarching decision structure.
        return model.guard.locking_atomic_node

    # All objects from this point on will return an atomic node.
    result = AtomicNode(model)

    if isinstance(model, Composite):
        # Create atomic nodes for each of the components, including the guard.
        atomic_nodes = [create_locking_structure(v) for v in [model.guard] + model.assignments]
        for n in atomic_nodes:
            result.include_atomic_node(n)

        # Chain all the components and connect the failure exit of the guard to the composite's atomic node.
        result.graph.add_edge(result.entry_node, atomic_nodes[0].entry_node)
        result.graph.add_edge(atomic_nodes[0].failure_exit, result.failure_exit)
        for i in range(1, len(atomic_nodes)):
            result.graph.add_edge(atomic_nodes[i - 1].success_exit, atomic_nodes[i].entry_node)
        result.graph.add_edge(atomic_nodes[-1].success_exit, result.success_exit)
    elif isinstance(model, Assignment):
        # The left hand side of the assignment cannot be locked locally, and hence will not get an atomic node.
        # The right side will only get an atomic node if it is a boolean expression or primary.
        if is_boolean_statement(model.right):
            # Create an atomic node and include it.
            right_atomic_node = create_locking_structure(model.right)

            # The assignment does not use the expression's partner evaluations. Hence, mark it as indifferent.
            right_atomic_node.mark_indifferent()

            # Add the right hand side atomic node to the graph.
            result.include_atomic_node(right_atomic_node)

            # Add  the appropriate connections.
            result.graph.add_edge(result.entry_node, right_atomic_node.entry_node)
            result.graph.add_edge(right_atomic_node.success_exit, result.success_exit)
        else:
            # Simply create a connection to the success exit point--the statement will be locked in its entirety.
            result.graph.add_edge(result.entry_node, result.success_exit)

        # Assignments cannot fail, and hence, the atomic node should be indifferent to the partner's evaluation.
        result.mark_indifferent()
    elif isinstance(model, Expression):
        # Conjunction and disjunction statements need special treatment due to their control flow characteristics.
        # Additionally, exclusive disjunction needs different treatment too, since it can have nested aggregates.
        if model.op in ["and", "or"]:
            # Find over which nodes the statement is made and add all the graphs to the result node.
            atomic_nodes = [create_locking_structure(v) for v in model.values]
            for n in atomic_nodes:
                result.include_atomic_node(n)

            # Connect all clauses that prematurely exit the expression to the appropriate exit point.
            for n in atomic_nodes:
                if model.op == "and":
                    result.graph.add_edge(n.failure_exit, result.failure_exit)
                else:
                    result.graph.add_edge(n.success_exit, result.success_exit)

            # Chain the remaining exit points and entry points in order of the clauses.
            result.graph.add_edge(result.entry_node, atomic_nodes[0].entry_node)
            for i in range(0, len(model.values) - 1):
                if model.op == "and":
                    result.graph.add_edge(atomic_nodes[i].success_exit, atomic_nodes[i + 1].entry_node)
                else:
                    result.graph.add_edge(atomic_nodes[i].failure_exit, atomic_nodes[i + 1].entry_node)
            if model.op == "and":
                result.graph.add_edge(atomic_nodes[-1].success_exit, result.success_exit)
            else:
                result.graph.add_edge(atomic_nodes[-1].failure_exit, result.failure_exit)
        elif model.op == "xor":
            # Find over which nodes the statement is made and add all the graphs to the result node.
            atomic_nodes = [create_locking_structure(v) for v in model.values]
            for n in atomic_nodes:
                # Nodes should be marked indifferent, since the partner's value does not alter the control flow.
                n.mark_indifferent()
                result.include_atomic_node(n)

            # Chain the exit points and entry points in order of the clauses.
            result.graph.add_edge(result.entry_node, atomic_nodes[0].entry_node)
            for i in range(0, len(model.values) - 1):
                result.graph.add_edge(atomic_nodes[i].success_exit, atomic_nodes[i + 1].entry_node)
            result.graph.add_edge(atomic_nodes[-1].success_exit, result.success_exit)
            result.graph.add_edge(atomic_nodes[-1].success_exit, result.failure_exit)
        else:
            # Add success and failure exit connections.
            # Note that math operators aren't treated differently--they cannot reach this point.
            result.graph.add_edge(result.entry_node, result.success_exit)
            result.graph.add_edge(result.entry_node, result.failure_exit)
    elif isinstance(model, Primary):
        if model.body is not None:
            # Add a child relationship to the body.
            child_node = create_locking_structure(model.body)

            result.include_atomic_node(child_node)
            result.graph.add_edge(result.entry_node, child_node.entry_node)

            if model.sign == "not":
                # Boolean negations simply switch the success and failure branch of the object in question.
                result.graph.add_edge(child_node.success_exit, result.failure_exit)
                result.graph.add_edge(child_node.failure_exit, result.success_exit)
            else:
                result.graph.add_edge(child_node.success_exit, result.success_exit)
                result.graph.add_edge(child_node.failure_exit, result.failure_exit)
        elif model.ref is not None:
            # Add a success exit connection. Additionally, add a failure connection of the variable is a boolean.
            result.graph.add_edge(result.entry_node, result.success_exit)
            if model.ref.var.is_boolean:
                result.graph.add_edge(result.entry_node, result.failure_exit)
        else:
            # The primary contains a constant.
            if model.ref is False:
                # Special case: don't connect the true branch, since the expression will always yield false.
                result.graph.add_edge(result.entry_node, result.failure_exit)
            elif model.ref is True:
                # Special case: don't connect the false branch, since the expression will always yield true.
                result.graph.add_edge(result.entry_node, result.success_exit)
            else:
                # For integer/byte values, there is only a success path.
                result.graph.add_edge(result.entry_node, result.success_exit)
    else:
        raise Exception("This node is not allowed to be part of the locking structure.")

    # Store the node inside the partner such that it can be accessed by the code generator.
    if isinstance(model, SlcoLockableNode):
        model.locking_atomic_node = result

    # Return the atomic node associated with the model.
    return result


def generate_base_level_locking_data(model: AtomicNode):
    """
    Add the requested lock objects to the base-level lockable components.
    """
    if isinstance(model.partner, Assignment):
        # Find the variables that are targeted by the assignment's atomic node.
        class_variable_references = get_class_variable_references(model.partner.left)

        if len(model.child_atomic_nodes) == 0:
            # No atomic node present for the right hand side. Add the requested locks to the assignment's atomic node.
            class_variable_references.update(get_class_variable_references(model.partner.right))
        else:
            # Recursively add the appropriate data to the right hand side.
            for n in model.partner.locking_atomic_node.child_atomic_nodes:
                generate_base_level_locking_data(n)

        # Create lock objects for all of the used class variables, and add them to the entry and exit points.
        locks = {Lock(r, model.entry_node) for r in class_variable_references}
        model.entry_node.locks_to_acquire.update(locks)
        model.success_exit.locks_to_release.update(locks)
        model.failure_exit.locks_to_release.update(locks)
    elif len(model.child_atomic_nodes) > 0:
        # The node is not a base-level lockable node. Continue recursively.
        for n in model.partner.locking_atomic_node.child_atomic_nodes:
            generate_base_level_locking_data(n)
    else:
        # The node is a base-level lockable node. Add the appropriate locks.
        class_variable_references = get_class_variable_references(model.partner)
        locks = {Lock(r, model.entry_node) for r in class_variable_references}
        model.entry_node.locks_to_acquire.update(locks)
        model.success_exit.locks_to_release.update(locks)
        model.failure_exit.locks_to_release.update(locks)


def restructure_lock_acquisitions(model: AtomicNode, nr_of_passes=2):
    """
    Move lock acquisitions upwards to the appropriate level in the locking graph.

    The process is done in passes--two might be needed to reach the desired effect if non-constant indices are present.
    """
    # Repeat the same process for the given number of passes.
    if nr_of_passes > 1:
        restructure_lock_acquisitions(model, nr_of_passes - 1)

    # Gather the locks that have been acquired prior to reaching the target node.
    locks_acquired_beforehand: Dict[LockingNode, Set[Lock]] = dict()
    n: LockingNode
    for n in nx.topological_sort(model.graph):
        locks_acquired_by_predecessors: Set[Lock] = set()
        p: LockingNode
        for p in model.graph.predecessors(n):
            # Add all of the previously seen locks by the targeted predecessor, with its own locks added as well.
            locks_acquired_by_predecessors.update(locks_acquired_beforehand[p])
            locks_acquired_by_predecessors.update(p.locks_to_acquire)

        # Add an entry for the current node.
        locks_acquired_beforehand[n] = locks_acquired_by_predecessors

    # Move locks that violate the ordering upwards until they are no longer in violation.
    n: LockingNode
    for n in reversed(list(nx.topological_sort(model.graph))):
        # Find the locks that have already been activated by all nodes occurring earlier in the graph.
        locks_acquired_by_predecessors: Set[Lock] = locks_acquired_beforehand[n]

        # Find the locks acquired in the current node that should be moved upwards.
        # A lock needs to be moved upwards if it should be requested before an already requested lock (<=).
        violating_lock_requests: Set[Lock] = {
            i for i in n.locks_to_acquire if any(i <= i2 for i2 in locks_acquired_by_predecessors)
        }

        # Move the violating locks upwards, while ensuring that the structure stays intact and sound.
        if len(violating_lock_requests) > 0:
            # Remove the violating locks from the current node.
            n.locks_to_acquire.difference_update(violating_lock_requests)

            # Add a rewrite rule when moving past the exit node of an assignment.
            if isinstance(n.partner, Assignment) and n.node_type.value > 1:
                for i in violating_lock_requests:
                    i.prepend_rewrite_rule((n.partner.left, n.partner.right))

            # Move the violating locks to all predecessors and ensure that locks are always released.
            p: LockingNode
            for p in model.graph.predecessors(n):
                # Add the locks to the predecessor's acquisition list.
                p.locks_to_acquire.update(violating_lock_requests)

                # Add lock releases if the target node has multiple successors to ensure that the locks are always
                # released regardless of the path taken.
                if len(list(model.graph.successors(p))) > 1:
                    q: LockingNode
                    for q in model.graph.successors(p):
                        q.locks_to_release.update(violating_lock_requests)


def restructure_lock_releases(model: AtomicNode):
    """
    Move locks downwards to the appropriate level in the locking graph.
    """
    # Gather the locks that have already been released by the nodes coming after the target node.
    locks_released_afterwards: Dict[LockingNode, Set[Lock]] = dict()
    n: LockingNode
    for n in reversed(list(nx.topological_sort(model.graph))):
        locks_released_by_successors: Set[Lock] = set()
        s: LockingNode
        for s in model.graph.successors(n):
            # Add all of the locks released after the target successor, with its own locks added as well.
            locks_released_by_successors.update(locks_released_afterwards[s])
            locks_released_by_successors.update(s.locks_to_release)

        # Add an entry for the current node.
        locks_released_afterwards[n] = locks_released_by_successors

    # Move locks that violate the ordering downwards until they are no longer in violation.
    n: LockingNode
    for n in nx.topological_sort(model.graph):
        # Find the locks that are released later on in the graph structure.
        locks_released_by_successors: Set[Lock] = locks_released_afterwards[n]

        # A lock needs to be moved down if a lock with the same target is released by a node further along in the graph.
        violating_lock_requests: Set[Lock] = {
            i for i in n.locks_to_release if any(i.ref == i2.ref for i2 in locks_released_by_successors)
        }

        # Move the violating locks downwards.
        # Note that it will generally imply that, if a lock is released further along in the graph, that an accompanying
        # lock request will also have been made beforehand. This lock request is moved upwards, and hence, it is assumed
        # for simplicity that locks do not have to be acquired upon merging nodes. Nevertheless, the validator should be
        # able to detect such violations occur.
        # TODO: write a validator.
        if len(violating_lock_requests) > 0:
            # Remove the violating locks from the current node.
            n.locks_to_release.difference_update(violating_lock_requests)

            # Move the violating locks to all successors.
            s: LockingNode
            for s in model.graph.successors(n):
                # Add the locks to the predecessor's release list.
                s.locks_to_release.update(violating_lock_requests)
