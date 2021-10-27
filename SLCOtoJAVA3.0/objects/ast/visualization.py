from typing import Dict

from networkx.drawing.nx_pydot import graphviz_layout

from locking.ordering import get_variable_ordering_graph
from objects.ast.models import Expression, Primary, VariableRef, Composite, Assignment, Class
import networkx as nx
import matplotlib.pyplot as plt

from objects.ast.util import get_variable_dependency_graph, get_weighted_variable_dependency_graph


def construct_graph(e, graph: nx.DiGraph, labels, parent_node=None):
    """Construct a graph for the given expression."""
    node_id = graph.number_of_nodes()
    if isinstance(e, Expression):
        labels[node_id] = e.op
        graph.add_node(node_id)
        if parent_node is not None:
            graph.add_edge(parent_node, node_id)
        for v in e.values:
            construct_graph(v, graph, labels, node_id)
    elif isinstance(e, Primary):
        if e.sign != "":
            labels[node_id] = e.sign
            graph.add_node(node_id)
            if parent_node is not None:
                graph.add_edge(parent_node, node_id)
            if e.body is not None:
                labels[node_id + 1] = "()"
                graph.add_node(node_id + 1)
                graph.add_edge(node_id, node_id + 1)
                construct_graph(e.body, graph, labels, node_id + 1)
            elif e.ref is not None:
                construct_graph(e.ref, graph, labels, node_id)
            else:
                labels[node_id + 1] = e.value
                graph.add_edge(node_id, node_id + 1)
                graph.add_node(node_id + 1)
        else:
            if e.body is not None:
                labels[node_id] = "()"
                graph.add_node(node_id)
                if parent_node is not None:
                    graph.add_edge(parent_node, node_id)
                construct_graph(e.body, graph, labels, node_id)
            elif e.ref is not None:
                construct_graph(e.ref, graph, labels, parent_node)
            else:
                labels[node_id] = e.value
                if parent_node is not None:
                    graph.add_edge(parent_node, node_id)
                graph.add_node(node_id)
    elif isinstance(e, VariableRef):
        if e.index is not None:
            labels[node_id] = e.var.name + "[]"
            graph.add_node(node_id)
            if parent_node is not None:
                graph.add_edge(parent_node, node_id)
            construct_graph(e.index, graph, labels, node_id)
        else:
            labels[node_id] = e.var.name
            graph.add_node(node_id)
            if parent_node is not None:
                graph.add_edge(parent_node, node_id)
    elif isinstance(e, Assignment):
        labels[node_id] = ":="
        graph.add_node(node_id)
        if parent_node is not None:
            graph.add_edge(parent_node, node_id)
        construct_graph(e.left, graph, labels, node_id)
        construct_graph(e.right, graph, labels, node_id)
    elif isinstance(e, Composite):
        labels[node_id] = "[;]"
        graph.add_node(node_id)
        if parent_node is not None:
            graph.add_edge(parent_node, node_id)
        construct_graph(e.guard, graph, labels, node_id)
        for v in e.assignments:
            construct_graph(v, graph, labels, node_id)


def visualize_expression(e):
    """Visualize the given expression as an AST plot."""
    graph = nx.DiGraph()
    labels = {}
    construct_graph(e, graph, labels)
    render_graph(graph, str(e) + " (AST)", labels)


def visualize_dependency_graph(e):
    """Visualize the dependency graph of the given expression."""
    graph = get_variable_dependency_graph(e)
    labels = {k: k.name for k in graph.nodes}
    render_graph(graph, str(e) + " (VDG)", labels)


def visualize_weighted_variable_dependency_graph(e: Class):
    """Visualize the dependency graph of the given class."""
    graph = get_weighted_variable_dependency_graph(e)
    labels = {k: k.name + "'" if k.is_class_variable else k.name for k in graph.nodes}
    render_graph(graph, str(e) + " (WVDG)", labels, layout="neato")


def visualize_variable_ordering_graph(e):
    """Visualize the dependency graph of the given expression."""
    graph = get_variable_ordering_graph(e)
    labels = {k: k.name for k in graph.nodes}
    render_graph(graph, str(e) + " (VOG)", labels)


def render_graph(graph: nx.DiGraph, title: str = "", labels: Dict = None, layout="dot"):
    """Visualize the given graph with the given data."""
    if len(graph.nodes) == 0:
        return

    # Determine the location and create an axis to put the title on.
    pos = graphviz_layout(graph, prog=layout)
    ax = plt.gca()
    ax.set_title(title)

    # Give self-loops a different color and remove self-loop edges.
    node_color = []
    for v in graph.nodes():
        if graph.has_edge(v, v):
            node_color.append("#FFCCCB")
            graph.remove_edge(v, v)
        else:
            node_color.append("#D0F0C0")

    # Draw the graph with the given colors.
    nx.draw(graph, pos, ax=ax, node_color=node_color)

    # Draw the labels if appropriate.
    if labels is not None:
        nx.draw_networkx_labels(graph, pos, labels)

    if len(nx.get_node_attributes(graph, "weight")) > 0:
        pos_attrs = {}
        # Add the weight as a label besides the node.
        for node, coords in pos.items():
            pos_attrs[node] = (coords[0], coords[1] + 12)
        node_attrs = nx.get_node_attributes(graph, "weight")
        custom_node_attrs = {}
        for node, attr in node_attrs.items():
            custom_node_attrs[node] = attr
        nx.draw_networkx_labels(graph, pos_attrs, labels=custom_node_attrs)

    if len(nx.get_edge_attributes(graph, "weight")) > 0:
        edge_attrs = nx.get_edge_attributes(graph, "weight")
        custom_edge_attrs = {}
        for edge, attr in edge_attrs.items():
            custom_edge_attrs[edge] = attr
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=custom_edge_attrs)

    ax.collections[0].set_edgecolor("#000000")

    if layout == "neato":
        ax.set_xlim([1.2 * x for x in ax.get_xlim()])
        ax.set_ylim([1.2 * y for y in ax.get_ylim()])

    plt.show()
