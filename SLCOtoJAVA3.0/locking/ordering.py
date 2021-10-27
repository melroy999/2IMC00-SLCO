import networkx as nx

from objects.ast.interfaces import SlcoStatementNode
from objects.ast.models import VariableRef, Expression, Primary, Composite
from objects.ast.util import __dfs__, get_variable_references


def get_variable_ordering_graph(model: SlcoStatementNode) -> nx.DiGraph:
    """Create a graph that contains the order in which variables occur in the statement."""
    # The ordering is equivalent to a DFS on the model.
    graph = nx.DiGraph()

    # Only evaluate guard expressions.
    if isinstance(model, Composite):
        model = model.guard

    if isinstance(model, (Expression, Primary)):
        encountered_variables = set()
        for e in __dfs__(model, self_first=True, _filter=lambda x: isinstance(x, VariableRef)):
            graph.add_node(e.var)
            if e.index is not None:
                # Add a relation to all encountered variables so far that also occur in the index.
                used_variables = {v.var for v in get_variable_references(e.index)}
                used_variables.intersection_update(encountered_variables)

                if len(used_variables) > 0:
                    # A variable proceeds another if it has an edge to its successor.
                    for v in used_variables:
                        graph.add_edge(v, e.var)

            # Add the variable to the list of encountered variables.
            encountered_variables.add(e.var)
    return graph
