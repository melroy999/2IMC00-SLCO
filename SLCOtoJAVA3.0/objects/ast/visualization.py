from networkx.drawing.nx_pydot import graphviz_layout

from objects.ast.models import Expression, Primary, VariableRef, Composite, Assignment
import networkx as nx
import matplotlib.pyplot as plt


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
    pos = graphviz_layout(graph, prog="dot")
    ax = plt.gca()
    ax.set_title(str(e))
    nx.draw(graph, pos, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels)
    plt.show()
