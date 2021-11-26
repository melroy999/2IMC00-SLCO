import logging
import math
from collections import defaultdict
from typing import Dict, Set

import networkx as nx

from objects.ast.interfaces import SlcoLockableNode
from objects.ast.models import VariableRef, Primary, Composite, Transition, Assignment, Expression, Variable
from objects.ast.util import get_class_variable_references
from objects.locking.models import AtomicNode, LockingNodeType, LockingNode
from objects.locking.visualization import render_locking_structure


def create_locking_structure(model: Transition) -> AtomicNode:
    logging.info(f"> Constructing the locking structure for object \"{model}\"")

    # Construct the locking structure for the object.
    result = construct_locking_structure(model)

    # Add information on what locks to lock at the base level.
    for s in model.statements:
        insert_base_level_lock_requests(s.locking_atomic_node)
        correct_lock_acquisitions(s.locking_atomic_node)
        correct_lock_acquisition_ordering(s.locking_atomic_node)
        correct_lock_releases(s.locking_atomic_node)
        render_locking_structure(s.locking_atomic_node)

    return result


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


def construct_locking_structure(model) -> AtomicNode:
    """
    Create a DAG of locking nodes that will dictate which locks will need to be requested/released at what position.

    The returned node is the encompassing atomic node of the model.
    """
    if isinstance(model, Transition):
        # The guard expression of the transition needs to be included in the main structure. Other statements do not.
        for s in model.statements:
            construct_locking_structure(s)

        # Chain the guard statement with the overarching decision structure.
        return model.guard.locking_atomic_node

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
        # The left hand side of the assignment cannot be locked locally, and hence will not get an atomic node.
        # The right side will only get an atomic node if it is a boolean expression or primary.
        if is_boolean_statement(model.right):
            # Create an atomic node and include it.
            right_atomic_node = construct_locking_structure(model.right)

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
    else:
        raise Exception("This node is not allowed to be part of the locking structure.")

    # Store the node inside the partner such that it can be accessed by the code generator.
    if isinstance(model, SlcoLockableNode):
        model.locking_atomic_node = result

    # Return the atomic node associated with the model.
    return result


def insert_base_level_lock_requests(model: AtomicNode):
    """
    Add lock request data to the appropriate locking nodes for all non-aggregate base-level nodes.
    """
    logging.debug(f"> Inserting base level locking requests into the atomic node of \"{model.partner}\"")
    n: LockingNode
    for n in model.graph.nodes:
        # Find the target object and determine what class variables should be locked/unlocked at the base level.
        target = n.partner
        target_variables = None
        if isinstance(target, Assignment):
            # The variable that is being assigned to never has an atomic node.
            target_variables = get_class_variable_references(target.left)

            if len(model.child_atomic_nodes) == 0:
                # The right hand side does not have an atomic node, and hence, all right-side locks need to be included.
                target_variables.update(get_class_variable_references(target.right))
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
    logging.debug(f"> Correcting duplicate lock acquisitions in the atomic node of \"{model.partner}\"")

    # Keep a mapping of locks that have already been opened by the target statement's predecessors and itself.
    logging.debug(f" - Gathering accumulated lock requests:")
    accumulated_lock_request: Dict[LockingNode, Set[VariableRef]] = dict()
    target: LockingNode
    for target in nx.topological_sort(model.graph):
        active_lock_requests: Set[VariableRef] = set(target.locks_to_acquire)
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
        violating_lock_requests: Set[VariableRef] = set()
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


def correct_lock_acquisition_ordering(model: AtomicNode):
    """
    Prematurely move lock requests of which it is certain that they will result in lock ordering conflicts, regardless
    of the to be assigned locks identities.

    For example:
        - x[1] cannot occur before x[0] in the locking structure, since id + 0 <= id + 1 will always hold.
        - If x[i] is included in the locking structure, all need to be requested at the same level since
          id + i <= id + j for all j in len(x) cannot be evaluated statically.

    Notes:
        - This method will only affect the placement of array type variables.
    """
    logging.info(
        f"> Correcting lock id independent lock acquisitions ordering issues in the atomic node of \"{model.partner}\""
    )
    # Track whether array variables have a non-constant index.
    variable_has_non_constant_indices = defaultdict(bool)
    target: LockingNode
    for target in model.graph.nodes:
        r: VariableRef
        for r in target.locks_to_acquire:
            if r.index is not None and isinstance(r.index, Primary):
                # Mark the variable as non constant if the index is not a constant valued primary.
                index: Primary = r.index
                if index.value is None:
                    variable_has_non_constant_indices[r.var] = True

    if any(variable_has_non_constant_indices.values()):
        logging.info(" - The following variables have non-constant indices:")
        for key, value in variable_has_non_constant_indices.items():
            if value:
                logging.info(f"   - {key}")
    else:
        logging.info(" - None of the values have non-constant indices")

    # Track the highest index of each variable that has been acquired by the target statement's predecessors and itself.
    logging.info(f" - Gathering accumulated lock requests:")
    accumulated_lock_request: Dict[LockingNode, Dict[Variable, float]] = dict()
    for target in nx.topological_sort(model.graph):
        active_lock_requests: Dict[Variable, float] = defaultdict(int)

        # Merge the accumulated data for the direct predecessor nodes.
        for n in model.graph.predecessors(target):
            for key, value in accumulated_lock_request[n].items():
                active_lock_requests[key] = max(value, active_lock_requests[key])

        # Add lock requests of the current node to the collection.
        r: VariableRef
        for r in target.locks_to_acquire:
            # Make the index infinite if the variable has a non-constant index to enforce acquisition at the top level.
            target_variable: Variable = r.var
            if variable_has_non_constant_indices[target_variable]:
                active_lock_requests[target_variable] = math.inf
            elif target_variable.is_array:
                target_index: Primary = r.index
                active_lock_requests[target_variable] = max(target_index.value, active_lock_requests[target_variable])

        # Add an entry for the current node.
        logging.info(f"   - {target.partner}.{target.node_type.name}: {active_lock_requests}")
        accumulated_lock_request[target] = dict(active_lock_requests)

    # Push the lock requests up to the correct level in the locking structure.
    logging.info(f" - Making ordering corrections:")
    for target in reversed(list(nx.topological_sort(model.graph))):
        # Find lock requests that conflict order wise with the accumulated lock requests of the predecessors.
        violating_lock_requests: Set[VariableRef] = set()
        active_lock_requests: Dict[Variable, float] = defaultdict(int)

        # Merge the accumulated data for the direct predecessor nodes.
        for n in model.graph.predecessors(target):
            for key, value in accumulated_lock_request[n].items():
                active_lock_requests[key] = max(value, active_lock_requests[key])

        # Find which nodes are in violation of the ordering, regardless of the assigned lock identities.
        for r in target.locks_to_acquire:
            # Note that non-array variables aren't present in the active_lock_requests list.
            if r.var in active_lock_requests:
                if isinstance(r.index, Primary) and r.index.value is not None:
                    if active_lock_requests[r.var] >= r.index.value:
                        violating_lock_requests.add(r)
                else:
                    # Non-constant index value. Movement is always necessary.
                    violating_lock_requests.add(r)

        # Move all violating locks upwards one level.
        if len(violating_lock_requests) > 0:
            # Remove the violating lock requests from the current node.
            logging.info(
                f"   - Node {target.partner}.{target.node_type.name} introduces order violating lock acquisitions"
            )
            target.locks_to_acquire.difference_update(violating_lock_requests)

            # Move the requests.
            for n in model.graph.predecessors(target):
                logging.info(
                    f"     - Moving lock requests {violating_lock_requests} from node "
                    f"\"{target.partner}.{target.node_type.name}\" to \"{n.partner}.{n.node_type.name}\""
                )
                n.locks_to_acquire.update(violating_lock_requests)

                # Add lock releases if the target node has multiple successors to ensure that the locks are always
                # released regardless of the path taken.
                if len(list(model.graph.successors(n))) > 1:
                    logging.info(
                        f"       - Movement into a node with multiple children. Adding additional locks to be"
                        f" released to the following nodes:"
                    )
                    for n2 in model.graph.successors(n):
                        n2.locks_to_release.update(violating_lock_requests)
                        logging.debug(f"       - {n2.partner}.{n2.node_type.name}")


def correct_lock_releases(model: AtomicNode):
    """
    Move lock request releases to the appropriate level in the locking graph to ensure atomicity of the statement.
    """
    logging.debug(f"> Correcting duplicate lock releases in the atomic node of \"{model.partner}\"")

    # Keep a mapping of locks that have already been released by the target statement's successors and itself.
    logging.debug(f" - Gathering accumulated lock requests:")
    accumulated_lock_request: Dict[LockingNode, Set[VariableRef]] = dict()
    target: LockingNode
    for target in reversed(list(nx.topological_sort(model.graph))):
        released_lock_requests: Set[VariableRef] = set(target.locks_to_release)
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
        violating_lock_requests: Set[VariableRef] = set()
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
