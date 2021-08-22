from typing import List, Set

import networkx as nx

from objects.ast.models import Class, Variable, VariableRef, Primary
from objects.ast.interfaces import SlcoStatementNode
from objects.ast.util import get_class_variable_references, get_class_variable_dependency_graph, \
    get_weighted_class_variable_dependency_graph
from util.graph import convert_to_directed_acyclic_graph


def assign_lock_identities(model: Class) -> None:
    """Assign lock identities to the global variables within the given class."""
    # Convert the graph to a DAG and assign lock identities to the class variables using topological sort.
    weighted_dependency_graph = get_weighted_class_variable_dependency_graph(model)
    directed_acyclic_graph = convert_to_directed_acyclic_graph(weighted_dependency_graph)
    reversed_graph = directed_acyclic_graph.reverse()

    v: Variable
    i = 0
    for v in nx.lexicographical_topological_sort(
            reversed_graph, key=lambda x: (-reversed_graph.nodes[x]["weight"], x.name)
    ):
        v.lock_id = i
        i += max(1, v.type.size)
        print(v, v.lock_id)


def get_lock_id_requests(model: SlcoStatementNode) -> List[VariableRef]:
    """Get the lock identities requested by the target object, sorted by lock identity with conflicts removed."""
    class_variable_references = get_class_variable_references(model)
    class_variable_dependency_graph = get_class_variable_dependency_graph(model)
    optimize_lock_requests(class_variable_references, class_variable_dependency_graph)
    unpack_violating_lock_requests(class_variable_references, class_variable_dependency_graph)
    print(model, "->", sorted(class_variable_references, key=lambda r: r.var.lock_id))
    return sorted(class_variable_references, key=lambda r: r.var.lock_id)


def optimize_lock_requests(lock_requests: Set[VariableRef], dependency_graph: nx.DiGraph) -> None:
    """Remove superfluous lock requests when appropriate and edit the graph to reflect these changes."""
    # Are all of the indices of an array variable locked statically?
    variables: List[Variable] = [v for v in dependency_graph.nodes if v.is_array]
    for v in variables:
        variable_refs = set(r for r in lock_requests if r.var == v)
        basic_refs = set(r for r in variable_refs if isinstance(r.index, Primary) and r.index.value is not None)
        if len(variable_refs) != len(basic_refs) and len(basic_refs) == v.type.size:
            # All values are covered. Remove all complex references and edit the graph.
            lock_requests.difference_update(variable_refs.difference(basic_refs))
            dependency_graph.remove_edges_from(list(dependency_graph.in_edges(v)))


def unpack_violating_lock_requests(lock_requests: Set[VariableRef], dependency_graph: nx.DiGraph) -> None:
    """Unpack lock requests that violate the strict ordering of locking and edit the graph to reflect these changes."""
    variables: List[Variable] = [v for v in dependency_graph.nodes if v.is_array]
    for v in variables:
        if any(v.lock_id <= n.lock_id for n in dependency_graph[v]):
            # Create the unpacked references, and replace the original values with the new ones.
            unpacked_references: Set[VariableRef] = set()
            for i in range(0, v.type.size):
                ref = VariableRef(var=v, index=Primary(target=i))
                unpacked_references.add(ref)

            # Remove all requests over v and replace them with unpacked references.
            lock_requests.difference_update([r for r in lock_requests if r.var == v])
            lock_requests.update(unpacked_references)

            # All indices are covered, and hence, v doesn't depend on other variables anymore. Remove v's dependencies.
            dependency_graph.remove_edges_from(list(dependency_graph.edges(v)))
