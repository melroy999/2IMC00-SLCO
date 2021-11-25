import logging
from typing import Dict, Set

import networkx as nx

from objects.ast.models import VariableRef, Primary, Composite, Transition, Assignment, Expression, LockRequest
from objects.ast.util import get_class_variable_references
from objects.locking.models import AtomicNode, LockingNodeType, LockingNode


# TODO: Transition is temporary.
def create_locking_structure(model: Transition) -> AtomicNode:
    logging.info(f"> Constructing the locking structure for object \"{model}\"")

    # Construct the locking structure for the object.
    result = construct_locking_structure(model)

    # Add information on what locks to lock at the base level.
    for s in model.statements:
        insert_base_level_lock_requests(s.locking_atomic_node)
        correct_lock_acquisitions(s.locking_atomic_node)
        correct_lock_releases(s.locking_atomic_node)
        # render_locking_structure(s.locking_atomic_node)

    return result


def construct_locking_structure(model) -> AtomicNode:
    """
    Create a DAG of locking nodes that will dictate which locks will need to be requested/released at what position.

    The returned node is the encompassing atomic node of the model.
    """
    if isinstance(model, Transition):
        # Chain the guard statement with the overarching decision structure.
        # TODO: Transition locking_atomic_node assignment is temporary.
        result = model.guard.locking_atomic_node = construct_locking_structure(model.guard)

        # The guard expression of the transition needs to be included in the main structure. Other statements do not.
        for s in model.statements[1:]:
            s.locking_atomic_node = construct_locking_structure(s)
        return result

    # All objects from this point on will return an atomic node.
    result = AtomicNode(model)

    if isinstance(model, Composite):
        # Create atomic nodes for each of the components, including the guard.
        atomic_nodes = [construct_locking_structure(v) for v in [model.guard] + model.assignments]
        for n in atomic_nodes:
            result.include_atomic_node(n)

        # Chain all the components and connect the failure exit of the guard to the composite's atomic node.
        result.graph.add_edge(result.entry_node, atomic_nodes[0].entry_node)
        result.graph.add_edge(atomic_nodes[0].failure_exit, result.failure_exit)
        for i in range(1, len(atomic_nodes)):
            result.graph.add_edge(atomic_nodes[i - 1].success_exit, atomic_nodes[i].entry_node)
        result.graph.add_edge(atomic_nodes[-1].success_exit, result.success_exit)
    elif isinstance(model, Assignment):
        # Locks are needed for both the target variable and the assigned expression simultaneously.
        left_atomic_node = construct_locking_structure(model.left)
        right_atomic_node = construct_locking_structure(model.right)

        # The assignment does not use the expression's partner evaluations. Hence, mark it as indifferent.
        right_atomic_node.mark_indifferent()

        # Add the nodes to the graph.
        result.include_atomic_node(left_atomic_node)
        result.include_atomic_node(right_atomic_node)

        # Chain the left atomic node to the right atomic node, assuming indifference.
        result.graph.add_edge(result.entry_node, left_atomic_node.entry_node)
        result.graph.add_edge(left_atomic_node.success_exit, right_atomic_node.entry_node)
        result.graph.add_edge(right_atomic_node.success_exit, result.success_exit)

        # Assignments cannot fail, and hence, the atomic node should be indifferent to the partner's evaluation.
        result.mark_indifferent()
    elif isinstance(model, Expression):
        # Conjunction and disjunction statements need special treatment due to their control flow characteristics.
        # Additionally, exclusive disjunction needs different treatment too, since it can have nested aggregates.
        if model.op in ["and", "or"]:
            # Find over which nodes the statement is made and add all the graphs to the result node.
            atomic_nodes = [construct_locking_structure(v) for v in model.values]
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
            atomic_nodes = [construct_locking_structure(v) for v in model.values]
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
            # Note that math operators aren't treated differently--they are marked indifferent in the assignment.
            result.graph.add_edge(result.entry_node, result.success_exit)
            result.graph.add_edge(result.entry_node, result.failure_exit)
    elif isinstance(model, Primary):
        if model.body is not None:
            # Add a child relationship to the body.
            child_node = construct_locking_structure(model.body)

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
    elif isinstance(model, VariableRef):
        # Reading a variable will always be successful--hence, mark indifferent.
        result.graph.add_edge(result.entry_node, result.success_exit)
        result.mark_indifferent()
    else:
        raise Exception("This situation has not yet been implemented.")

    # Return the atomic node associated with the model.
    return result


def insert_base_level_lock_requests(model: AtomicNode):
    """
    Add lock request data to the appropriate locking nodes for all non-aggregate base-level nodes.
    """
    logging.debug(f"> Inserting base level locking requests into the atomic node of object \"{model.partner}\"")
    n: LockingNode
    for n in model.graph.nodes:
        # Find the target object and determine what class variables should be locked/unlocked at the base level.
        target = n.partner
        target_variables = None
        if isinstance(target, VariableRef):
            target_variables = get_class_variable_references(target)
        elif isinstance(target, Primary) and target.ref is not None:
            target_variables = get_class_variable_references(target)
        elif isinstance(target, Expression) and target.op not in ["and", "or", "xor"]:
            target_variables = get_class_variable_references(target)

        if target_variables is not None:
            if n.node_type == LockingNodeType.ENTRY:
                n.locks_to_acquire.update(target_variables)
                logging.debug(
                    f" - \"{n.partner}.{n.node_type.name}\".locks_to_acquire = {n.locks_to_acquire}"
                )
            else:
                n.locks_to_release.update(target_variables)
                logging.debug(
                    f" - \"{n.partner}.{n.node_type.name}\".locks_to_release = {n.locks_to_release}"
                )


def correct_lock_acquisitions(model: AtomicNode):
    """
    Move lock request acquisitions to the appropriate level in the locking graph to ensure atomicity of the statement.
    """
    logging.debug(f"> Correcting duplicate lock acquisitions in the atomic node of object \"{model.partner}\"")

    # Keep a mapping of locks that have already been opened by the target statement's predecessors and itself.
    logging.debug(f" - Gathering accumulated lock requests:")
    accumulated_lock_request: Dict[LockingNode, Set[LockRequest]] = dict()
    target: LockingNode
    for target in nx.topological_sort(model.graph):
        active_lock_requests: Set[LockRequest] = set(target.locks_to_acquire)
        for n in model.graph.predecessors(target):
            # Add all of the previously seen locks by the targeted predecessor.
            active_lock_requests.update(accumulated_lock_request[n])

        # Add an entry for the current node.
        logging.debug(f"   - {target.partner}.{target.node_type.name}: {active_lock_requests}")
        accumulated_lock_request[target] = active_lock_requests

    # Next, iterate over the structure with a reverse topological ordering.
    # Move locks that have already been requested by predecessor nodes to all of the node's predecessors.
    logging.debug(f" - Making ordering corrections:")
    for target in reversed(list(nx.topological_sort(model.graph))):
        # Find lock requests that intersect with any of the accumulated lock requests of the predecessors.
        violating_lock_requests = set()
        for n in model.graph.predecessors(target):
            violating_lock_requests.update(target.locks_to_acquire.intersection(accumulated_lock_request[n]))

        # Move all violating locks upwards one level.
        if len(violating_lock_requests) > 0:
            # Remove the violating lock requests from the current node.
            logging.debug(f"   - Node {target.partner}.{target.node_type.name} introduces duplicate lock acquisitions")
            target.locks_to_acquire.difference_update(violating_lock_requests)

            # Move the requests.
            for n in model.graph.predecessors(target):
                logging.debug(
                    f"     - Moving lock requests {violating_lock_requests} from node "
                    f"\"{target.partner}.{target.node_type.name}\" to \"{n.partner}.{n.node_type.name}\""
                )
                n.locks_to_acquire.update(violating_lock_requests)


def correct_lock_releases(model: AtomicNode):
    """
    Move lock request releases to the appropriate level in the locking graph to ensure atomicity of the statement.
    """
    logging.debug(f"> Correcting duplicate lock releases in the atomic node of object \"{model.partner}\"")

    # Keep a mapping of locks that have already been released by the target statement's successors and itself.
    logging.debug(f" - Gathering accumulated lock requests:")
    accumulated_lock_request: Dict[LockingNode, Set[LockRequest]] = dict()
    target: LockingNode
    for target in reversed(list(nx.topological_sort(model.graph))):
        released_lock_requests: Set[LockRequest] = set(target.locks_to_release)
        for n in model.graph.successors(target):
            # Add all of the previously seen locks by the targeted predecessor.
            released_lock_requests.update(accumulated_lock_request[n])

        # Add an entry for the current node.
        logging.debug(f"   - {target.partner}.{target.node_type.name}: {released_lock_requests}")
        accumulated_lock_request[target] = released_lock_requests

    # Next, iterate over the structure with a topological ordering.
    # Move locks that have already been released by successor nodes to all of the node's successors.
    logging.debug(f" - Making ordering corrections:")
    for target in nx.topological_sort(model.graph):
        # Find lock requests that intersect with any of the accumulated lock requests of the successors.
        violating_lock_requests = set()
        for n in model.graph.successors(target):
            violating_lock_requests.update(target.locks_to_release.intersection(accumulated_lock_request[n]))

        # Move all violating locks upwards one level.
        if len(violating_lock_requests) > 0:
            # Remove the violating lock requests from the current node.
            logging.debug(f"   - Node {target.partner}.{target.node_type.name} introduces duplicate lock releases")
            target.locks_to_release.difference_update(violating_lock_requests)

            # Move the requests.
            for n in model.graph.successors(target):
                logging.debug(
                    f"     - Moving lock requests {violating_lock_requests} from node "
                    f"\"{target.partner}.{target.node_type.name}\" to \"{n.partner}.{n.node_type.name}\""
                )
                n.locks_to_release.update(violating_lock_requests)
