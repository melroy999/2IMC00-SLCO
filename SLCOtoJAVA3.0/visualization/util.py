from typing import Callable, Dict

import networkx as nx
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout


def render_graph(
        graph: nx.DiGraph, title: str = "", layout="dot", node_color_func: Callable = None, labels: Dict = None
):
    """Visualize the given graph with the given data."""
    if len(graph.nodes) == 0:
        return

    plt.figure(1, figsize=(28, 20))

    # Determine the location and create an axis to put the title on.
    pos = graphviz_layout(graph, prog=layout)
    ax = plt.gca()
    ax.set_title(title)

    # Give self-loops a different color and remove self-loop edges.
    node_color = []
    if node_color is not None:
        for v in graph.nodes():
            node_color.append(node_color_func(v))

    # Draw the graph with the given colors.
    nx.draw(graph, pos, ax=ax, node_color=node_color)

    # Draw the labels if appropriate.
    if labels is not None:
        for i in pos:  # raise text positions
            pos[i] = (pos[i][0], pos[i][1] + 35)  # probably small value enough
        nx.draw_networkx_labels(graph, pos, labels)

    ax.collections[0].set_edgecolor("#000000")

    if layout == "neato":
        ax.set_xlim([1.2 * x for x in ax.get_xlim()])
        ax.set_ylim([1.2 * y for y in ax.get_ylim()])

    plt.show()
