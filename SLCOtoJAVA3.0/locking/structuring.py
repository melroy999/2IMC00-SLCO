import logging
from typing import Dict, Set, List, Union

import networkx as nx

import settings
from locking.validation import validate_locking_structure_integrity
from objects.ast.interfaces import SlcoLockableNode, SlcoStatementNode
from objects.ast.models import Primary, Composite, Transition, Assignment, Expression, StateMachine, Class, \
    VariableRef, Variable, DecisionNode, State
from objects.ast.util import get_variable_references, get_variables_to_be_locked
from objects.locking.models import AtomicNode, LockingNode, Lock, LockRequest, LockRequestInstanceProvider
from objects.locking.visualization import render_locking_structure_instructions


def initialize_transition_locking_structure(model: Transition):
    """
    Construct a locking structure for the targeted transition.
    """
    # Construct the locking structure for the transition.
    create_transition_locking_structure(model)

    # Generate the base-level locking data and add it to the locking structure.
    for s in model.statements:
        generate_base_level_locking_entries(s.locking_atomic_node)
        assign_locking_node_ids(s.locking_atomic_node)
        generate_location_sensitivity_checks(s.locking_atomic_node)
        generate_unavoidable_location_sensitivity_violation_marks(s.locking_atomic_node)

    # Perform a movement step prior to knowing the lock identities.
    # Two passes are required: one to get the non-constant array variables to the right position in the graph, and a
    # second to compensate for the effect that the movement of the aforementioned non-constant array indices may have
    # on the location of constant valued indices.
    for s in model.statements:
        restructure_lock_acquisitions(s.locking_atomic_node, nr_of_passes=2)


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


def construct_composite_node(model: Composite, result: AtomicNode) -> None:
    """
    Add components to the DAG of locking nodes for the given composite node.
    """
    # Create atomic nodes for each of the components, including the guard.
    atomic_nodes = [create_transition_locking_structure(v) for v in [model.guard] + model.assignments]
    for n in atomic_nodes:
        result.include_atomic_node(n)

    # Chain all the components and connect the failure exit of the guard to the composite's atomic node.
    result.graph.add_edge(result.entry_node, atomic_nodes[0].entry_node)
    result.graph.add_edge(atomic_nodes[0].failure_exit, result.failure_exit)
    for i in range(1, len(atomic_nodes)):
        result.graph.add_edge(atomic_nodes[i - 1].success_exit, atomic_nodes[i].entry_node)
    result.graph.add_edge(atomic_nodes[-1].success_exit, result.success_exit)


def construct_assignment_node(model: Assignment, result: AtomicNode) -> None:
    """
    Add components to the DAG of locking nodes for the given assignment node.
    """
    # The left hand side of the assignment cannot be locked locally, and hence will not get an atomic node.
    # The right side will only get an atomic node if it is a boolean expression or primary.
    if is_boolean_statement(model.right):
        # Create an atomic node and include it.
        right_atomic_node = create_transition_locking_structure(model.right)

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


def construct_expression_node(model: Expression, result: AtomicNode) -> None:
    """
    Add components to the DAG of locking nodes for the given expression node.
    """
    # Conjunction and disjunction statements need special treatment due to their control flow characteristics.
    # Additionally, exclusive disjunction needs different treatment too, since it can have nested aggregates.
    if model.op in ["and", "or"]:
        # Find over which nodes the statement is made and add all the graphs to the result node.
        atomic_nodes = [create_transition_locking_structure(v) for v in model.values]
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
        atomic_nodes = [create_transition_locking_structure(v) for v in model.values]
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


def construct_primary_node(model: Primary, result: AtomicNode) -> None:
    """
    Add components to the DAG of locking nodes for the given primary node.
    """
    if model.body is not None:
        # Add a child relationship to the body.
        child_node = create_transition_locking_structure(model.body)

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
        result.graph.add_edge(result.entry_node, result.success_exit)
        result.graph.add_edge(result.entry_node, result.failure_exit)


def create_transition_locking_structure(model: Union[Transition, SlcoStatementNode]) -> AtomicNode:
    """
    Create a DAG of locking nodes that will dictate which locks will need to be requested/released at what position.
    """
    if isinstance(model, Transition):
        # The guard expression of the transition needs to be included in the main structure. Other statements do not.
        for s in model.statements:
            create_transition_locking_structure(s)
        return model.locking_atomic_node

    # All objects from this point on will return an atomic node.
    result = AtomicNode(model)

    if isinstance(model, Composite):
        construct_composite_node(model, result)
    elif isinstance(model, Assignment):
        construct_assignment_node(model, result)
    elif isinstance(model, Expression):
        construct_expression_node(model, result)
    elif isinstance(model, Primary):
        construct_primary_node(model, result)
    else:
        raise Exception("This node is not allowed to be part of the locking structure.")

    # Store the node inside the partner such that it can be accessed by the code generator.
    if isinstance(model, SlcoLockableNode):
        model.locking_atomic_node = result

    # Return the atomic node associated with the model.
    return result


def generate_base_level_locking_entries(model: AtomicNode):
    """
    Add the requested lock objects to the base-level lockable components.
    """
    if isinstance(model.partner, Assignment):
        # Find the variables that are targeted by the assignment's atomic node.
        class_variable_references = get_variables_to_be_locked(model.partner.left)

        if len(model.child_atomic_nodes) == 0:
            # No atomic node present for the right hand side. Add the requested locks to the assignment's atomic node.
            class_variable_references.update(get_variables_to_be_locked(model.partner.right))
        else:
            # Recursively add the appropriate data to the right hand side.
            for n in model.partner.locking_atomic_node.child_atomic_nodes:
                generate_base_level_locking_entries(n)
                model.used_variables.update(n.used_variables)

        # Create lock objects for all of the used class variables, and add them to the entry and exit points.
        locks = {Lock(r, model.entry_node) for r in class_variable_references}
        model.entry_node.locks_to_acquire.update(locks)
        model.success_exit.locks_to_release.update(locks)
        model.failure_exit.locks_to_release.update(locks)
        model.used_variables.update(r.var for r in class_variable_references)
    elif len(model.child_atomic_nodes) > 0:
        # The node is not a base-level lockable node. Continue recursively.
        for n in model.partner.locking_atomic_node.child_atomic_nodes:
            generate_base_level_locking_entries(n)
            model.used_variables.update(n.used_variables)
    else:
        # The node is a base-level lockable node. Add the appropriate locks.
        class_variable_references = get_variables_to_be_locked(model.partner)
        locks = {Lock(r, model.entry_node) for r in class_variable_references}
        model.entry_node.locks_to_acquire.update(locks)
        model.success_exit.locks_to_release.update(locks)
        model.failure_exit.locks_to_release.update(locks)
        model.used_variables.update(r.var for r in class_variable_references)


def get_bound_checked_variable_references(model: SlcoStatementNode) -> Set[VariableRef]:
    """
    Get the referenced variables that are potentially bound checked by the given statement.
    """
    # This function should only be called for base-level lockable components.
    assert(model.locking_atomic_node is None or len(model.locking_atomic_node.child_atomic_nodes) == 0)

    result: Set[VariableRef] = set()
    if isinstance(model, VariableRef):
        # Note that only the variable itself is range checked, and not its indices.
        # Moreover, boolean types cannot be bound checked, since they cannot be used in the index of a variable.
        if not model.var.is_boolean:
            result.add(model)
    else:
        for v in model:
            result.update(get_bound_checked_variable_references(v))

    # Return all the used variables.
    return result


def get_bound_checked_variables(model: AtomicNode) -> Set[Lock]:
    """
    Get the referenced variables that are potentially bound checked by the given statement.
    """
    # This function should only be called for base-level lockable components.
    assert(len(model.child_atomic_nodes) == 0)

    # Get all the variable references that are bound checked.
    bound_checked_variable_references = get_bound_checked_variable_references(model.partner)

    # Find all lock objects that are bound checked.
    result: Set[Lock] = set()
    for i in model.entry_node.locks_to_acquire:
        if i.ref in bound_checked_variable_references:
            result.add(i)

    # Return all the used variables.
    return result


def generate_location_sensitivity_checks(model: AtomicNode, aggregate_variable_references: Set[Lock] = None):
    """
    Add verification pointers to locks that depend upon the control flow for a successful/error prone evaluation.
    """
    # Check which locks have been encountered so far.
    if aggregate_variable_references is None:
        aggregate_variable_references: Set[Lock] = set()

    # Perform a DFS on the objects atomic nodes.
    if len(model.child_atomic_nodes) > 0:
        # Generate the checks for all children.
        for n in model.child_atomic_nodes:
            generate_location_sensitivity_checks(n, aggregate_variable_references)
            model.location_sensitive_locks.extend(n.location_sensitive_locks)
    else:
        # The node is a base-level lockable component. Check if the node uses variables that have been encountered
        # previously in the index of class array variables.
        array_class_variable_references: Set[Lock] = {
            r for r in model.entry_node.locks_to_acquire if r.ref.var.is_array
        }

        # Check for each to be acquired locks if it is position sensitive or not.
        # A lock is location sensitive if its index uses a variable that is constrained/used earlier in the atomic tree.
        i: Lock
        for i in array_class_variable_references:
            # Get the variables used in the index of the array variable.
            variable_references = get_variable_references(i.ref.index)

            # The lock is position sensitive if any of the references are in the aggregated list.
            bound_checked_index_variables: List[Lock] = []
            j: Lock
            for j in aggregate_variable_references:
                if j.ref in variable_references:
                    bound_checked_index_variables.append(j)
            if len(bound_checked_index_variables) > 0:
                i.is_location_sensitive = True
                i.bound_checked_index_variables = bound_checked_index_variables
                model.location_sensitive_locks.append(i)
                logging.info(
                    f"Marking {i}.{i.original_locking_node.id} as location sensitive with bound checked indices "
                    f"[{', '.join(f'{v}.{v.original_locking_node.id}' for v in bound_checked_index_variables)}]."
                )

        # Add all the used variables to the aggregate variables list.
        aggregate_variable_references.update(get_bound_checked_variables(model))


def generate_unavoidable_location_sensitivity_violation_marks(model: AtomicNode, encountered_locks: Set[Lock] = None):
    # Check which locks have been encountered so far.
    if encountered_locks is None:
        encountered_locks: Set[Lock] = set()

    # Do a depth-first search.
    if len(model.child_atomic_nodes) > 0:
        # Generate the checks for all children.
        for n in model.child_atomic_nodes:
            generate_unavoidable_location_sensitivity_violation_marks(n, encountered_locks)
    else:
        # The node is a base-level lockable component.
        location_sensitive_locks = [i for i in model.entry_node.locks_to_acquire if i.is_location_sensitive]
        for i in location_sensitive_locks:
            # Find all locks already encountered with the same variable.
            variable_sharing_locks = [j for j in encountered_locks if i.ref.var == j.ref.var]
            if len(variable_sharing_locks):
                first_variable_occurrence = min((j.original_locking_node.id for j in variable_sharing_locks))

                # Check if the lock with the shared variable occurs before a bound checked index variable.
                if any(
                    j.original_locking_node.id >= first_variable_occurrence for j in i.bound_checked_index_variables
                ):
                    i.unavoidable_location_conflict = True
                    logging.info(
                        f"Flagging lock {i}.{i.original_locking_node.id} due to the occurrence of an unavoidable "
                        f"location sensitivity violation caused by one of the following locks: "
                        f"[{', '.join(f'{v}.{v.original_locking_node.id}' for v in variable_sharing_locks)}]."
                    )

    # Add all the used variables to the encountered variables list.
    encountered_locks.update(model.entry_node.locks_to_acquire)


def initialize_main_locking_structure(model: StateMachine, state: State):
    """
    Construct a locking structure for the targeted decision structure.
    """
    # Assert that no locking identities have been assigned yet.
    _class: Class = model.parent
    assert(len(_class.variables) == 0 or all(v.lock_id == -1 for v in _class.variables))

    # Find the root of the target decision structure.
    if state not in model.state_to_decision_node:
        return
    decision_structure_root = model.state_to_decision_node[state]

    # Construct the locking structure for the structure.
    create_main_locking_structure(decision_structure_root)
    assign_locking_node_ids(decision_structure_root.locking_atomic_node)

    # Perform a movement step prior to knowing the lock identities.
    # Two passes are required: one to get the non-constant array variables to the right position in the graph, and a
    # second to compensate for the effect that the movement of the aforementioned non-constant array indices may have
    # on the location of constant valued indices.
    restructure_lock_acquisitions(decision_structure_root.locking_atomic_node, nr_of_passes=2)


def get_max_list_size(model: AtomicNode) -> int:
    """
    Get the maximum number of locks that will be targeted at any time by the acquire/release locks methods.
    """
    max_size: int = 0
    n: LockingNode
    for n in nx.topological_sort(model.graph):
        instructions = n.locking_instructions
        max_size = max(
            max_size,
            max((len(phase) for phase in instructions.locks_to_acquire_phases), default=0),
            len(instructions.unpacked_lock_requests),
            len(instructions.locks_to_release)
        )
        pass

    return max_size


def construct_deterministic_decision_node(model: DecisionNode, result: AtomicNode) -> None:
    """
    Add components to the DAG of locking nodes for the given deterministic decision node.
    """
    # Get an atomic node for all of the options.
    atomic_nodes = [create_main_locking_structure(v) for v in model.decisions]
    for n in atomic_nodes:
        result.include_atomic_node(n)

    # When one branch fails, the control flow will proceed to the next one in line.
    result.graph.add_edge(result.entry_node, atomic_nodes[0].entry_node)
    for i in range(1, len(atomic_nodes)):
        result.graph.add_edge(atomic_nodes[i - 1].failure_exit, atomic_nodes[i].entry_node)
    result.graph.add_edge(atomic_nodes[-1].failure_exit, result.failure_exit)


def construct_non_deterministic_decision_node(model: DecisionNode, result: AtomicNode) -> None:
    """
    Add components to the DAG of locking nodes for the given non-deterministic decision node.
    """
    # Get an atomic node for all of the options.
    atomic_nodes = [create_main_locking_structure(v) for v in model.decisions]
    for n in atomic_nodes:
        result.include_atomic_node(n)

    # Given that the decision is completely random, the entry is connected to the entry of all choices.
    # Moreover, each failure exit is connected to the failure exit of the node.
    for n in atomic_nodes:
        result.graph.add_edge(result.entry_node, n.entry_node)
        result.graph.add_edge(n.failure_exit, result.failure_exit)


def construct_sequential_decision_node(model: DecisionNode, result: AtomicNode) -> None:
    """
    Add components to the DAG of locking nodes for the given sequential decision node.
    """
    # Get an atomic node for all of the options.
    atomic_nodes = [create_main_locking_structure(v) for v in model.decisions]
    for n in atomic_nodes:
        result.include_atomic_node(n)

    # Having the sequential operation be completely atomic would be a large hit on the performance.
    # Hence, use the same approach taken for non-determinism.
    for n in atomic_nodes:
        result.graph.add_edge(result.entry_node, n.entry_node)
        result.graph.add_edge(n.failure_exit, result.failure_exit)


def create_main_locking_structure(model) -> AtomicNode:
    """
    Create a DAG of locking nodes that will dictate which locks will need to be requested/released at what position.

    The returned node is the encompassing atomic node of the model.
    """
    if isinstance(model, Transition):
        # The guard expression of the transition needs to be included in the main structure. Other statements do not.
        # Chain the guard statement with the overarching decision structure.
        return model.locking_atomic_node

    # All objects from this point on will return an atomic node.
    result = AtomicNode(model)

    if isinstance(model, DecisionNode):
        # The structure depends on the type of decision made.
        if model.is_deterministic:
            construct_deterministic_decision_node(model, result)
        elif settings.non_determinism:
            construct_non_deterministic_decision_node(model, result)
        else:
            if settings.atomic_sequential:
                # The sequential should behave like a deterministic group instead.
                construct_deterministic_decision_node(model, result)
            else:
                construct_sequential_decision_node(model, result)
    else:
        raise Exception("This node is not allowed to be part of the locking structure.")

    # Store the node inside the partner such that it can be accessed by the code generator.
    if isinstance(model, SlcoLockableNode):
        model.locking_atomic_node = result

    # Return the atomic node associated with the model.
    return result


def assign_locking_node_ids(model: AtomicNode):
    """
    Assign numbers to the locking nodes such that the node's predecessors always have a lower id.
    """
    n: LockingNode
    for n in nx.topological_sort(model.graph):
        # Get the maximum id of the node's predecessors.
        n.id = max((m.id + 1 for m in model.graph.predecessors(n)), default=0)


def finalize_locking_structure(model: StateMachine, state: State):
    """
    Finalize the locking structure for the given decision structure based on the given lock priorities.
    """
    # Assert that the locking identities have been assigned.
    _class: Class = model.parent
    assert(len(_class.variables) == 0 or any(v.lock_id != -1 for v in _class.variables))

    # Find the root of the target decision structure.
    if state not in model.state_to_decision_node:
        return
    decision_structure_root = model.state_to_decision_node[state]

    # Not all atomic nodes are part of the main decision structure. Find the objects that need to be iterated over.
    target_atomic_nodes = [decision_structure_root.locking_atomic_node]
    for t in model.state_to_transitions[state]:
        target_atomic_nodes.extend([s.locking_atomic_node for s in t.statements[1:]])

    # Perform another restructuring pass.
    for n in target_atomic_nodes:
        restructure_lock_acquisitions(n)

    # Find unresolvable violations and mark them as dirty.
    for n in target_atomic_nodes:
        generate_dirty_lock_marks(n)

    # Create the locking instructions. Create a lock request instance provider for the state specifically.
    lock_request_instance_provider = LockRequestInstanceProvider()
    for n in target_atomic_nodes:
        generate_locking_instructions(n, lock_request_instance_provider)

    # Move the lock releases to the appropriate location.
    for n in target_atomic_nodes:
        restructure_lock_releases(n)

    # Validate the locking structure.
    for n in target_atomic_nodes:
        validate_locking_structure_integrity(n)

    # Save allocation data for the locking parameters.
    model.lock_ids_list_size = max((get_max_list_size(n) for n in target_atomic_nodes), default=0)
    model.target_locks_list_size = max(model.target_locks_list_size, lock_request_instance_provider.counter)

    # Render the locking structure as an image.
    if settings.visualize_locking_graph:
        for n in target_atomic_nodes:
            render_locking_structure_instructions(n)


def restructure_lock_acquisitions(model: AtomicNode, nr_of_passes=2):
    """
    Move lock acquisitions upwards to the appropriate level in the locking graph.

    The process is done in passes--two might be needed to reach the desired effect if non-constant indices are present.
    The non-constant index reference needs to first be raised to the appropriate level for constant indices to follow.
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


def generate_dirty_lock_marks(model: AtomicNode):
    """
    Mark the locks that violate the desired lock ordering or structural behavior as dirty.
    """
    # Check for location sensitivity issues.
    mark_location_sensitivity_violations(model)

    # Mark array variables that violate the strict lock ordering as dirty.
    mark_lock_ordering_violations(model)


def mark_location_sensitivity_violations(model: AtomicNode):
    """
    Find location sensitive locks that have been moved and mark them as dirty.
    """
    # Iterate over all the locking nodes and search for locks in the acquisition list that are location sensitive and
    # are no longer part of their original locking node.
    n: LockingNode
    for n in model.graph.nodes:
        i: Lock
        for i in (j for j in n.locks_to_acquire if j.is_location_sensitive):
            # Check if the lock has moved past or into the node of one of its bound checked variables.
            if any(n.id <= j.original_locking_node.id for j in i.bound_checked_index_variables):
                # The lock has been moved. Mark as dirty.
                i.is_dirty = True
                logging.info(
                    f"Marking lock {i} in \"{model.partner}\" as dirty due to a location sensitivity violation. The "
                    f"lock moved from node {i.original_locking_node.partner}.{i.original_locking_node.node_type.name} "
                    f"to {n.partner}.{n.node_type.name}."
                )


def mark_lock_ordering_violations(model: AtomicNode):
    """
    Find array class variables that use variables in the index that have not been locked prior to reading.
    """
    # Check for every array variable that is acquired for lock ordering violations in the index.
    n: LockingNode
    for n in model.graph.nodes:
        i: Lock
        for i in (j for j in n.locks_to_acquire if j.ref.var.is_array):
            class_variable_references: Set[VariableRef] = get_variables_to_be_locked(i.ref.index)
            if any(r for r in class_variable_references if r.var.lock_id >= i.ref.var.lock_id):
                # A violation is found. Mark the lock as dirty.
                i.is_dirty = True
                logging.info(f"Marking lock {i} in \"{model.partner}\" as dirty due to a lock ordering violation.")


def get_unpacked_lock_requests(i: Lock, provider: LockRequestInstanceProvider) -> Set[LockRequest]:
    """
    Get the variable references needed to lock the entirety of the array associated with the lock's target variable.
    """
    target_references = {VariableRef(i.ref.var, Primary(target=j)) for j in range(i.ref.var.type.size)}
    return {LockRequest.get(r, provider) for r in target_references}


def generate_locking_phases(lock_requests: Set[LockRequest]) -> List[List[LockRequest]]:
    """
    Split the given list of lock requests into the appropriate locking phases.
    """
    # TODO: Simplified to having a phase for each individual variable. Certain phases can be merged.
    # Each variable gets a separate phase, with the phases being ordered by lock identity.
    variable_groupings: Dict[Variable, List[LockRequest]] = dict()
    for r in lock_requests:
        variable_groupings[r.var] = variable_groupings.get(r.var, [])
        variable_groupings[r.var].append(r)

    # Create the phases in the order of the lock identities of the primary variables.
    result: List[List[LockRequest]] = []
    variables: List[Variable] = list(variable_groupings.keys())
    for v in sorted(variables, key=lambda x: x.lock_id):
        result.append(variable_groupings[v])

    return result


def generate_locking_instructions(model: AtomicNode, provider: LockRequestInstanceProvider):
    """
    Add the appropriate entries in the locking instruction objects referenced with the locking nodes.
    """
    # Check for every array variable that is acquired for lock ordering violations in the index.
    n: LockingNode
    for n in nx.topological_sort(model.graph):
        # The instructions object of the lock in question.
        instructions = n.locking_instructions

        # Find the locks that are considered safe to attain under the lock ordering and locality restrictions.
        safe_to_be_acquired_locks = {i for i in n.locks_to_acquire if not i.is_dirty}
        instructions.locks_to_acquire.update(LockRequest.get(i, provider) for i in safe_to_be_acquired_locks)

        # Find the locks that need to be unpacked.
        dirty_to_be_acquired_locks = n.locks_to_acquire.difference(safe_to_be_acquired_locks)

        # Perform weak or strict unpacking.
        i: Lock
        for i in dirty_to_be_acquired_locks:
            supplemental_lock_requests = get_unpacked_lock_requests(i, provider)
            if i.is_location_sensitive:
                # Strict unpacking. The entire array will need to stay locked until the violating lock is released.
                instructions.locks_to_acquire.update(supplemental_lock_requests)
            else:
                # Weak unpacking. Variable references used exclusively in unpacking can be let go of at the end.
                # Note that the supplemental lock releases are moved by another algorithm to the appropriate spot.
                instructions.locks_to_acquire.update(supplemental_lock_requests)
                instructions.unpacked_lock_requests.add(LockRequest.get(i, provider))
                instructions.locks_to_release.update(supplemental_lock_requests)

        # Generate the locking phases.
        instructions.locks_to_acquire_phases = generate_locking_phases(instructions.locks_to_acquire)

        # Next, process the lock releases. Ensure that the proper replacements are added for strict unpacking.
        strictly_unpacked_to_be_released_locks = {
            i for i in n.locks_to_release if i.is_dirty and i.is_location_sensitive
        }
        safe_to_be_released_locks = n.locks_to_release.difference(strictly_unpacked_to_be_released_locks)

        # Gather release data and perform weak or strict unpacking.
        instructions.locks_to_release.update(LockRequest.get(i, provider) for i in safe_to_be_released_locks)
        i: Lock
        for i in strictly_unpacked_to_be_released_locks:
            supplemental_lock_requests = get_unpacked_lock_requests(i, provider)
            instructions.locks_to_release.update(supplemental_lock_requests)


def restructure_lock_releases(model: AtomicNode):
    """
    Move lock requests downwards to the appropriate level in the locking graph's locking instructions.
    """
    # Gather the lock requests that have already been released by the nodes coming after the target node.
    locks_released_afterwards: Dict[LockingNode, Set[LockRequest]] = dict()
    n: LockingNode
    for n in reversed(list(nx.topological_sort(model.graph))):
        locks_released_by_successors: Set[LockRequest] = set()
        s: LockingNode
        for s in model.graph.successors(n):
            # Add all of the locks released after the target successor, with its own locks added as well.
            locks_released_by_successors.update(locks_released_afterwards[s])
            locks_released_by_successors.update(s.locking_instructions.locks_to_release)

        # Add an entry for the current node.
        locks_released_afterwards[n] = locks_released_by_successors

    # Move locks that violate the ordering downwards until they are no longer in violation.
    n: LockingNode
    for n in nx.topological_sort(model.graph):
        # Find the locks that are released later on in the graph structure.
        locks_released_by_successors: Set[LockRequest] = locks_released_afterwards[n]

        # A lock needs to be moved down if a lock with the same target is released by a node further along in the graph.
        violating_lock_requests: Set[LockRequest] = n.locking_instructions.locks_to_release.intersection(
            locks_released_by_successors
        )

        # Move the violating locks downwards.
        # Note that it will generally imply that, if a lock is released further along in the graph, that an accompanying
        # lock request will also have been made beforehand. This lock request is moved upwards, and hence, it is assumed
        # for simplicity that locks do not have to be acquired upon merging nodes. Nevertheless, the validator should be
        # able to detect such violations occur.
        if len(violating_lock_requests) > 0:
            # Remove the violating locks from the current node.
            n.locking_instructions.locks_to_release.difference_update(violating_lock_requests)

            # Move the violating locks to all successors.
            s: LockingNode
            for s in model.graph.successors(n):
                # Add the locks to the predecessor's release list.
                s.locking_instructions.locks_to_release.update(violating_lock_requests)

    # Gather the requires and ensures data needed by the VerCors verification.
    n: LockingNode
    for n in nx.topological_sort(model.graph):
        # The instructions object of the lock in question.
        instructions = n.locking_instructions

        # Find which locks are active prior to reaching the node by looking at the data of previous nodes.
        # Note that all predecessors should have the same locks active for the locking graph to be correct.
        p: LockingNode
        for p in model.graph.predecessors(n):
            instructions.requires_lock_requests.update(p.locking_instructions.ensures_lock_requests)
            break

        # Find which lock requests will be active after execution of the locking instruction.
        instructions.ensures_lock_requests.update(instructions.requires_lock_requests)
        instructions.ensures_lock_requests.update(instructions.locks_to_acquire)
        instructions.ensures_lock_requests.update(instructions.unpacked_lock_requests)
        instructions.ensures_lock_requests.difference_update(instructions.locks_to_release)
