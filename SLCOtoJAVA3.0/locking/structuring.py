from objects.ast.models import VariableRef, Primary, Composite, Transition, Assignment, Expression
from objects.locking.models import AtomicNode
from objects.locking.visualization import render_locking_structure


def create_locking_structure(model):
    """
    Create a DAG of locking nodes that will dictate which locks will need to be requested/released at what position.

    The returned node is the encompassing atomic node of the model.
    """
    if isinstance(model, Transition):
        # Chain the guard statement with the overarching decision structure.
        result = create_locking_structure(model.guard)

        # The guard expression of the transition needs to be included in the main structure. Other statements do not.
        for s in model.statements[1:]:
            create_locking_structure(s)
        return result

    # All objects from this point on will return an atomic node.
    result = AtomicNode(model)

    if isinstance(model, Composite):
        # Create atomic nodes for each of the components, including the guard.
        atomic_nodes = [create_locking_structure(v) for v in [model.guard] + model.assignments]
        for n in atomic_nodes:
            result.include_atomic_node(n)

        # Chain all the components and connect the failure exit of the guard to the composite's atomic node.
        result.graph.add_edge(result.entry_node, atomic_nodes[0].entry_node)
        for i in range(0, len(atomic_nodes) - 1):
            result.graph.add_edge(atomic_nodes[i].success_exit, atomic_nodes[i + 1].entry_node)
        result.graph.add_edge(atomic_nodes[-1].success_exit, result.success_exit)
        result.graph.add_edge(atomic_nodes[0].failure_exit, result.failure_exit)

        render_locking_structure(result)
    elif isinstance(model, Assignment):
        # Locks are needed for both the target variable and the assigned expression simultaneously.
        left_atomic_node = create_locking_structure(model.left)
        right_atomic_node = create_locking_structure(model.right)

        # An assignment cannot fail. Hence, the two components should be indifferent in terms of evaluation result.
        left_atomic_node.make_indifferent()
        right_atomic_node.make_indifferent()

        # Add the nodes to the graph.
        result.include_atomic_node(left_atomic_node)
        result.include_atomic_node(right_atomic_node)

        # Chain the left atomic node to the right atomic node.
        result.graph.add_edge(result.entry_node, left_atomic_node.entry_node)
        result.graph.add_edge(left_atomic_node.success_exit, right_atomic_node.entry_node)
        result.graph.add_edge(right_atomic_node.success_exit, result.success_exit)
        result.graph.add_edge(right_atomic_node.failure_exit, result.success_exit)

        # Mark all the nodes in the right side as passive nodes if not boolean--locks cannot be requested at this level.
        if not model.left.var.is_boolean:
            right_atomic_node.mark_as_passive_recursively()
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

            # Locks should be acquired and released in the aggregate's clauses.
            result.mark_as_passive()
        elif model.op == "xor":
            # Find over which nodes the statement is made and add all the graphs to the result node.
            atomic_nodes = [create_locking_structure(v) for v in model.values]

            # Make all the atomic nodes indifferent and include them.
            for n in atomic_nodes:
                n.make_indifferent()
                result.include_atomic_node(n)

            # Chain the exit points and entry points in order of the clauses.
            result.graph.add_edge(result.entry_node, atomic_nodes[0].entry_node)
            for i in range(0, len(model.values) - 1):
                result.graph.add_edge(atomic_nodes[i].success_exit, atomic_nodes[i + 1].entry_node)
                result.graph.add_edge(atomic_nodes[i].failure_exit, atomic_nodes[i + 1].entry_node)
            result.graph.add_edge(atomic_nodes[-1].success_exit, result.success_exit)
            result.graph.add_edge(atomic_nodes[-1].failure_exit, result.success_exit)
            result.graph.add_edge(atomic_nodes[-1].success_exit, result.failure_exit)
            result.graph.add_edge(atomic_nodes[-1].failure_exit, result.failure_exit)

            # Locks should be acquired and released in the aggregate's clauses.
            result.mark_as_passive()
        elif model.op in ["==", "<>", "<=", ">=", "<", ">"]:
            # Boolean operators connect the entry point to both the exit points.
            result.graph.add_edge(result.entry_node, result.success_exit)
            result.graph.add_edge(result.entry_node, result.failure_exit)
        else:
            # Other expressions have one entry node, connected to a success exit node.
            result.graph.add_edge(result.entry_node, result.success_exit)

            # Mark all the nodes in the result as passive nodes--locks cannot be requested at this level.
            result.mark_as_passive_recursively()
    elif isinstance(model, Primary):
        # Find the node of the target object within the primary.
        if model.body is not None:
            child_node = create_locking_structure(model.body)
        elif model.ref is not None:
            child_node = create_locking_structure(model.ref)
        else:
            child_node = create_locking_structure(model.value)

        result.include_atomic_node(child_node)
        result.graph.add_edge(result.entry_node, child_node.entry_node)

        if model.sign == "not":
            # Boolean negations simply switch the success and failure branch of the object in question.
            result.graph.add_edge(child_node.success_exit, result.failure_exit)
            result.graph.add_edge(child_node.failure_exit, result.success_exit)
        else:
            result.graph.add_edge(child_node.success_exit, result.success_exit)
            result.graph.add_edge(child_node.failure_exit, result.failure_exit)
    elif isinstance(model, VariableRef):
        # Depending on the type of the variable, add either an edge from the entry to the successful exit, or both.
        result.graph.add_edge(result.entry_node, result.success_exit)
        if model.var.is_boolean:
            result.graph.add_edge(result.entry_node, result.failure_exit)

        # Mark all the nodes in the result as passive nodes--locks cannot be requested at this level.
        result.mark_as_passive_recursively()
    else:
        if model is False:
            # Connect only to the failure branch.
            result.graph.add_edge(result.entry_node, result.failure_exit)
        else:
            # Connect only to the success branch.
            result.graph.add_edge(result.entry_node, result.success_exit)

        # Mark all the nodes in the result as passive nodes--locks cannot be requested at this level.
        result.mark_as_passive_recursively()

    # Return the atomic node associated with the model.
    return result


def remove_repeating_lock_requests(model: AtomicNode):
    """
    Remove lock requests in the locking structure that have been acquired by an earlier statement already.
    """
    # Use a dictionary, since we do not want the data to remain in the data structure.
    # Iterate over the graph with topological ordering and check for the locks already acquired by the parents.
    # Filter out locks that are already acquired by the parent nodes.
    pass
