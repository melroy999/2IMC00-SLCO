import networkx as nx
import matplotlib.pyplot as plt

from objects.ast.models import SLCOModel


def draw(model: SLCOModel):
    # https://networkx.org/documentation/stable/reference/algorithms/index.html
    for c in model.classes:
        composition = c.get_weighted_class_variable_dependency_graph()
        render_graph(composition)

        acyclic = convert_to_directed_acyclic_graph(composition)
        render_graph(acyclic)

        reverse = acyclic.reverse(copy=True)
        render_graph(reverse)

        ordering = [(v, v.lock_id) for v in nx.topological_sort(reverse)]
        print(ordering)


def convert_to_directed_acyclic_graph(input_graph: nx.DiGraph, target_nodes: bool = False) -> nx.DiGraph:
    """Convert the given graph object to an acyclic graph by strategically removing edges until the graph is a DAG."""
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
                # If multiple candidates, evaluate the weight of the weight of the target node.
                target_edge = min(
                    graph.edges(), key=lambda u: (graph[u[0]][u[1]]["weight"], graph.nodes[u[1]]["weight"])
                )
                graph.remove_edge(*target_edge)

    if not nx.is_directed_acyclic_graph(graph):
        raise Exception("Failed to turn the given graph into a DAG.")

    return graph


def render_graph(graph: nx.DiGraph, has_weights: bool = True) -> None:
    """Render the given graph for visual feedback."""
    if len(graph.nodes) == 0:
        return

    plt.subplot(111)
    pos_nodes = nx.spiral_layout(graph)
    node_color = []

    # for each node in the graph
    for v in graph.nodes():
        if graph.has_edge(v, v):
            node_color.append("#FFCCCB")
        else:
            node_color.append("#D0F0C0")
    nx.draw(graph, pos_nodes, with_labels=True, node_color=node_color)

    if has_weights:
        pos_attrs = {}
        for node, coords in pos_nodes.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.12)
        node_attrs = nx.get_node_attributes(graph, "weight")
        custom_node_attrs = {}
        for node, attr in node_attrs.items():
            custom_node_attrs[node] = attr
        nx.draw_networkx_labels(graph, pos_attrs, labels=custom_node_attrs)
        edge_attrs = nx.get_edge_attributes(graph, "weight")

        custom_edge_attrs = {}
        for edge, attr in edge_attrs.items():
            custom_edge_attrs[edge] = attr
        nx.draw_networkx_edge_labels(graph, pos_nodes, edge_labels=custom_edge_attrs)

    axis = plt.gca()
    axis.collections[0].set_edgecolor("#000000")
    axis.set_xlim([1.3 * x for x in axis.get_xlim()])
    axis.set_ylim([1.3 * y for y in axis.get_ylim()])
    plt.show()
