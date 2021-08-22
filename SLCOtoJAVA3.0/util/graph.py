import networkx as nx


def convert_to_directed_acyclic_graph(input_graph: nx.DiGraph, target_nodes: bool = False) -> nx.DiGraph:
    """Convert the given graph object to an acyclic graph by strategically removing edges until the graph is a DAG."""
    # Create a copy to avoid altering the original.
    graph = input_graph.copy()
    nodes_to_process = set(graph.nodes)

    # First, remove all self cycles.
    graph.remove_edges_from([(v, v) for v in nodes_to_process])

    while len(nodes_to_process) > 0:
        # Pick a node that has no outgoing edges (topological sort with the edge direction reversed).
        leaf_nodes = [n for n in nodes_to_process if len([v for v in graph[n].keys() if v in nodes_to_process]) == 0]
        if len(leaf_nodes) > 0:
            nodes_to_process.difference_update(leaf_nodes)
        else:
            # Remove a node or edge and make another attempt.
            if target_nodes:
                # Find the lowest weight node and remove all of its still unresolved outgoing edges.
                target = min(nodes_to_process, key=lambda v: graph.nodes[v]["weight"])
                target_edges = [(target, n) for n in [v for v in graph[target].keys() if v in nodes_to_process]]
                graph.remove_edges_from(target_edges)
            else:
                # Find the lowest weight unresolved edge and remove it.
                # If multiple candidates, evaluate the weight of the target node.
                target_edge = min(
                    graph.edges(), key=lambda u: (graph[u[0]][u[1]]["weight"], graph.nodes[u[1]]["weight"])
                )
                graph.remove_edge(*target_edge)

    # Throw an exception if the revised graph still contains cycles.
    if not nx.is_directed_acyclic_graph(graph):
        raise Exception("Failed to turn the given graph into a DAG.")

    return graph
