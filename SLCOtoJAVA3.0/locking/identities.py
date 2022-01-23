from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Set, Tuple

import networkx as nx

from objects.ast.models import Class, Variable, VariableRef
from objects.ast.util import get_class_variable_references

if TYPE_CHECKING:
    from objects.locking.models import AtomicNode, LockingNode


def assign_lock_identities(model: Class) -> None:
    """Assign lock identities to the global variables within the given class."""
    generate_lock_identities(model)


def assign_lock_identity(v: Variable, i: int) -> int:
    """
    Assign a lock identity to the variable and increment the lock counter.
    """
    v.lock_id = i
    i += max(1, v.type.size)
    logging.info(f" - {v} {v.lock_id}")
    return i


def normalize_heuristic_targets(graph: nx.DiGraph):
    """
    Add normalized variants of the variables used by the heuristics when appropriate.
    """
    # Keep totals of all target variables.
    total_ls_t_weight = 0
    total_ls_weight = 0
    total_is_location_sensitive = 0
    total_an_weight = 0
    for _, data in graph.nodes(data=True):
        total_ls_t_weight += data["ls:t_weight"]
        total_ls_weight += data["ls:weight"]
        total_is_location_sensitive += data["ls:is_location_sensitive"]
        total_an_weight += data["an:weight"]

    # Normalize based on the totals.
    for v, data in graph.nodes(data=True):
        data["ls:t_weight_n"] = data["ls:t_weight"] / max(1, total_ls_t_weight)
        data["ls:weight_n"] = data["ls:weight"] / max(1, total_ls_weight)
        data["ls:is_location_sensitive_n"] = data["ls:is_location_sensitive"] / max(1, total_is_location_sensitive)
        data["an:weight_n"] = data["an:weight"] / max(1, total_an_weight)

    # Keep a total of all edge weights starting in the node in question.
    e_total_t_ir_weight = 0
    e_total_ir_weight = 0
    e_total_ls_weight = 0
    for v, _, data in graph.edges(data=True):
        e_total_t_ir_weight += data["ir:t_weight"]
        e_total_ir_weight += data["ir:weight"]
        e_total_ls_weight += data["ls:weight"]

    # Normalize based on the totals.
    for v, _, data in graph.edges(data=True):
        data["ir:t_weight_n"] = data["ir:t_weight"] / max(1, e_total_t_ir_weight)
        data["ir:weight_n"] = data["ir:weight"] / max(1, e_total_ir_weight)
        data["ls:weight_n"] = data["ls:weight"] / max(1, e_total_ls_weight)
    #
    # for e in graph.nodes(data=True):
    #     print(e)
    # for e in graph.edges(data=True):
    #     print(e)


def get_sortable_heuristic_tuple(v: Variable, graph: nx.DiGraph) -> Tuple:
    """
    Express the heuristic as a tuple, in which the items are ordered on their overall importance.
    """
    # Observations:
    #   1. Location sensitive variables have the largest impact on the locking and concurrency when unpacked.
    #   2. Unpacking operations will only occur for array variables. Hence, prefer selecting them over others.
    #   3. Unpacking needs to always occur when a self-loop is present. Hence, give them a lower id.
    #   4a. The more a variable is used, the greater the effect of its unpacking.
    #   4b. The more variables proceed another in the locking graph, the more likely it is that unpacking will occur if
    #   the lock is given a lower identity.
    #   5. As a last resort, use the variable name to make a decision.
    # data = graph.nodes[v]
    # node_weights = data["ls:t_weight_n"] + data["ls:weight_n"] + data["an:weight_n"]
    # edge_weights = 0
    # for w in graph.successors(v):
    #     data_w = graph.edges[v, w]
    #     edge_weights += data_w["ir:t_weight_n"] + data_w["ir:weight_n"] + data_w["ls:weight_n"]
    # return data["ls:is_location_sensitive"], v.is_array, graph.has_edge(v, v), node_weights + edge_weights, v.name
    #
    #

    data = graph.nodes[v]
    weighted_total = data["ls:is_location_sensitive_n"]
    weighted_total += 0 if v.is_array else 2
    weighted_total += data["ls:t_weight_n"]
    weighted_total += data["ls:weight_n"]
    weighted_total += data["an:weight_n"]
    for w in graph.successors(v):
        data_w = graph.edges[v, w]
        weighted_total += data_w["ir:t_weight_n"]
        weighted_total += data_w["ir:weight_n"]
        weighted_total += data_w["ls:weight_n"]

    return weighted_total, v.name




def generate_lock_identities(model: Class) -> None:
    # Construct the dependency graph based on the generated locking graph.
    graph = construct_weighted_class_variable_dependency_graph(model)

    # Continue until the graph is empty.
    i = 0
    while len(graph) > 0:
        # Find nodes that have no dependencies until no such nodes can be found.
        while True:
            variables_with_no_dependencies: Set[Variable] = set(v for v in graph if len(set(graph.successors(v))) == 0)

            # Process the nodes and remove them from the graph.
            for v in sorted(
                variables_with_no_dependencies, key=lambda x: (graph.nodes[x]["ls:is_location_sensitive"], x.name)
            ):
                i = assign_lock_identity(v, i)
                graph.remove_node(v)

            # Stop once no more variables can be found without dependencies.
            if len(variables_with_no_dependencies) == 0:
                break

        # If variables still remain, select and extract a variable according to certain heuristics.
        if len(graph) > 0:
            target_variables = list(graph)

            # Select a variable based on the selection heuristics.
            normalize_heuristic_targets(graph)
            target_variables.sort(key=lambda v: get_sortable_heuristic_tuple(v, graph))
            target_variable = target_variables[0]
            i = assign_lock_identity(target_variable, i)
            graph.remove_node(target_variable)


def get_initial_node_fields():
    """The initial fields of nodes added to the graph."""
    return {
        "ls:t_weight": 0,
        "ls:weight": 0,
        "an:weight": 0,
        "ls:is_location_sensitive": 0
    }


def get_initial_edge_fields():
    """The initial fields of edges added to the graph."""
    return {
        "ls:weight": 0,
        "ir:weight": 0,
        "ir:t_weight": 0
    }


def construct_weighted_class_variable_dependency_graph(model: Class) -> nx.DiGraph:
    """
    Construct a weighted dependency graph for the class variables used within the model.
    """
    # Create the graph.
    graph = nx.DiGraph()

    # Insert all the class variables.
    for v in model.variables:
        graph.add_node(v, **get_initial_node_fields())

    for sm in model.state_machines:
        for s, dn in sm.state_to_decision_node.items():
            # Gather the atomic nodes to gather dependency information from.
            target_atomic_nodes = [dn.locking_atomic_node]
            for t in sm.state_to_transitions[s]:
                target_atomic_nodes.extend(s.locking_atomic_node for s in t.statements[1:])

            # Process the relations of each of the target nodes.
            for n in target_atomic_nodes:
                process_locking_sub_graph(n, graph)

    return graph


def increment_variable_weight(v: Variable, field: str, graph: nx.DiGraph):
    """
    Increment the given field for a node in the graph.
    """
    if not graph.has_node(v):
        graph.add_node(v, **get_initial_node_fields())
    graph.nodes[v][field] += 1


def add_variable_weight(v: Variable, weight: int, field: str, graph: nx.DiGraph):
    """
    Set the given field for a node in the graph to true.
    """
    if not graph.has_node(v):
        graph.add_node(v, **get_initial_node_fields())
    graph.nodes[v][field] += weight


def increment_edge_weight(_from: Variable, _to: Variable, field: str, graph: nx.DiGraph):
    """
    Increment the given field for an edge in the graph.
    """
    if not graph.has_edge(_from, _to):
        graph.add_edge(_from, _to, **get_initial_edge_fields())
    graph[_from][_to][field] += 1


def process_locking_sub_graph(model: AtomicNode, graph: nx.DiGraph) -> None:
    """
    Add the relations found within the sub-graph to the weighted dependency graph.
    """
    # Track which variables have been used in the locking structure.
    encountered_variables: Set[Variable] = set()

    # Gather the variables that have been encountered before reaching the node in question.
    variables_encountered_beforehand: Dict[LockingNode, Set[Variable]] = dict()
    n: LockingNode
    for n in nx.topological_sort(model.graph):
        # Find all variables that have been encountered already earlier in the locking structure.
        variables_visited_by_predecessors: Set[Variable] = set()
        p: LockingNode
        for p in model.graph.predecessors(n):
            # Add all of the previously seen locks by the targeted predecessor, with its own locks added as well.
            variables_visited_by_predecessors.update(variables_encountered_beforehand[p])

        # Find all variables used by the locking node and apply the weights.
        variables_used_in_locking_node = [r.ref.var for r in n.locks_to_acquire]
        v: Variable
        for v in variables_used_in_locking_node:
            increment_variable_weight(v, "ls:t_weight", graph)
        for v in set(variables_used_in_locking_node):
            increment_variable_weight(v, "ls:weight", graph)

        # Mark a node as location sensitive if appropriate.
        for i in n.locks_to_acquire:
            if i.is_location_sensitive and not i.unavoidable_location_conflict:
                # Avoid adding weight to the location sensitivity if a violation is unavoidable.
                add_variable_weight(i.ref.var, len(model.graph), "ls:is_location_sensitive", graph)

        # Create the dependencies between variables.
        variable_references: Set[VariableRef] = set(r.ref for r in n.locks_to_acquire)
        variable_dependencies: Set[Tuple[Variable, Variable]] = set()
        for source in variable_references:
            # Create a dependency to all variables encountered earlier in the locking graph.
            v: Variable
            for v in variables_visited_by_predecessors:
                # Note that variable references sharing the same variable are already in a valid ordering by the
                # construction of the locking graph.
                if source.var != v:
                    increment_edge_weight(source.var, v, "ls:weight", graph)

            # Add potential internal relations.
            if source.index is not None:
                target_references = get_class_variable_references(source.index)
                for target in target_references:
                    if (source.var, target.var) not in variable_dependencies:
                        variable_dependencies.add((source.var, target.var))

                        # Add the edge to the graph and update the weight.
                        increment_edge_weight(source.var, target.var, "ir:weight", graph)

                    # Add the edge to the graph and update the weight.
                    increment_edge_weight(source.var, target.var, "ir:t_weight", graph)

        # Add an entry for the current node.
        visited_variables = set(r.var for r in variable_references)
        encountered_variables.update(visited_variables)
        variables_encountered_beforehand[n] = variables_visited_by_predecessors.union(visited_variables)

    # Add the atomic node level weight for the variables.
    v: Variable
    for v in encountered_variables:
        increment_variable_weight(v, "an:weight", graph)
