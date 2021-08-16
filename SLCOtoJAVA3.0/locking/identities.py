from typing import Set, List
import networkx as nx

import objects.ast.models as models
import objects.ast.interfaces as interfaces
from objects.util.graph import convert_to_directed_acyclic_graph, render_graph

# Type abbreviations
SLCOModel = models.SlcoModel
VariableRef = models.VariableRef
Variable = models.Variable
Primary = models.Primary
Expression = models.Expression
Lockable = interfaces.Lockable


def assign_lock_identities(model: SLCOModel) -> None:
    """Assign lock identities to the given model."""
    for c in model.classes:
        weighted_graph = c.get_weighted_class_variable_dependency_graph()
        render_graph(weighted_graph)
        weighted_dag = convert_to_directed_acyclic_graph(weighted_graph)
        render_graph(weighted_dag)
        reversed_weighted_dag = weighted_dag.reverse()

        # Reverse the arrow direction for the desired topological ordering.
        v: Variable
        i = 0
        for v in nx.topological_sort(reversed_weighted_dag):
            v.lock_id = i
            i += max(1, v.type.size)
            print(v, v.lock_id)
        render_graph(reversed_weighted_dag)


def gather_lock_ids(model: SLCOModel):
    for c in model.classes:
        for sm in c.state_machines:
            for g in sm.state_to_transitions.values():
                for t in g:
                    for s in t.statements:
                        requests = get_lock_id_requests(s)


def get_lock_id_requests(target: Lockable) -> List[VariableRef]:
    """Get the lock identities requested by the target object, sorted by lock identity."""
    variable_references = target.get_class_variable_references()
    dependency_graph = target.get_class_variable_dependency_graph()
    optimize_lock_requests(variable_references, dependency_graph)
    unpack_violating_lock_requests(variable_references, dependency_graph)
    print(target, "->", sorted(variable_references, key=lambda r: r.var.lock_id))
    return sorted(variable_references, key=lambda r: r.var.lock_id)


def unpack_violating_lock_requests(lock_requests: Set[VariableRef], dependency_graph: nx.DiGraph) -> None:
    """Unpack lock requests that violate the strict ordering of locking and edit the graph to reflect these changes."""
    variables: Set[Variable] = set(v for v in dependency_graph.nodes if v.is_array)
    for v in variables:
        if any(v.lock_id <= n.lock_id for n in dependency_graph[v]):
            # Create the unpacked references, and replace the original values with the new ones.
            unpacked_references: Set[VariableRef] = set()
            for i in range(0, v.type.size):
                ref = VariableRef()
                ref._var = v
                ref.index = Primary("", i)
                unpacked_references.add(ref)

            # Remove all requests over v and replace them with unpacked references.
            lock_requests.difference_update([r for r in lock_requests if r.var == v])
            lock_requests.update(unpacked_references)

            # All indices are covered, and hence, v doesn't depend on other variables anymore. Remove v's dependencies.
            dependency_graph.remove_edges_from(list(dependency_graph.edges(v)))


def optimize_lock_requests(lock_requests: Set[VariableRef], dependency_graph: nx.DiGraph) -> None:
    """Remove superfluous lock requests when appropriate and edit the graph to reflect these changes."""
    # Are all of the indices of an array variable locked statically?
    variables: Set[Variable] = set(v for v in dependency_graph.nodes if v.is_array)
    for v in variables:
        variable_refs = set(r for r in lock_requests if r.var == v)
        basic_refs = set(r for r in variable_refs if isinstance(r.index, Primary) and r.index.value is not None)
        if len(variable_refs) != len(basic_refs) and len(basic_refs) == v.type.size:
            # All values are covered. Remove all complex references and edit the graph.
            lock_requests.difference_update(variable_refs.difference(basic_refs))
            dependency_graph.remove_edges_from(list(dependency_graph.in_edges(v)))
