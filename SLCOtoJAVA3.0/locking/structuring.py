import networkx as nx

from objects.ast.models import VariableRef, Primary, Composite, Transition, Assignment, Expression
from objects.ast.util import get_class_variable_references
from objects.locking.models import AtomicNode, LockingNodeType
from objects.locking.visualization import render_locking_structure


# TODO: Transition is temporary.
def create_locking_structure(model: Transition) -> AtomicNode:
    # Construct the locking structure for the object.
    result = construct_locking_structure(model)

    # Add information on what locks to lock at the base level.
    for s in model.statements:
        insert_base_level_lock_requests(s.locking_atomic_node)
        remove_repeating_lock_requests(s.locking_atomic_node)
        remove_repeating_lock_releases(s.locking_atomic_node)
        render_locking_structure(s.locking_atomic_node)

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
    for n in model.graph.nodes:
        # Find the target object and determine what class variables should be locked/unlocked at the base level.
        target = n.partner
        if isinstance(target, VariableRef):
            n.target_locks.update(get_class_variable_references(target))
        elif isinstance(target, Primary) and target.ref is not None:
            n.target_locks.update(get_class_variable_references(target))
        elif isinstance(target, Expression) and target.op not in ["and", "or", "xor"]:
            n.target_locks.update(get_class_variable_references(target))


def remove_repeating_lock_requests(model: AtomicNode):
    """
    Remove lock requests in the locking structure that have been acquired by an earlier statement already.
    """
    # Track which locks have been activated by a specific node and its parents.
    activated_lock_requests = dict()

    # Iterate through the graph in topological ordering to ensure that predecessors have all the required data.
    for target in nx.topological_sort(model.graph):
        active_lock_requests = set()
        for n in model.graph.predecessors(target):
            active_lock_requests.update(activated_lock_requests[n])

        if target.node_type == LockingNodeType.ENTRY:
            # Remove the locks that are already active.
            target.target_locks.difference_update(active_lock_requests)
            active_lock_requests.update(target.target_locks)

        # Add an entry for the current node.
        activated_lock_requests[target] = active_lock_requests


def remove_repeating_lock_releases(model: AtomicNode):
    """
    Remove lock releases in the locking structure that are released prematurely with respect to its children's targets.
    """
    # Track which locks have been activated by a specific node and its parents.
    closed_lock_requests = dict()

    # Iterate through the graph in reversed topological ordering to ensure that successors have all the required data.
    for target in reversed(list(nx.topological_sort(model.graph))):
        released_lock_requests = set()
        for n in model.graph.successors(target):
            released_lock_requests.update(closed_lock_requests[n])

        if target.node_type in [LockingNodeType.SUCCESS, LockingNodeType.FAILURE]:
            # Remove the locks that are to be released later on.
            target.target_locks.difference_update(released_lock_requests)
            released_lock_requests.update(target.target_locks)

        # Add an entry for the current node.
        closed_lock_requests[target] = released_lock_requests
