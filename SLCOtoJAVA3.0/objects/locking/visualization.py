from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from objects.locking.models import LockingNode, AtomicNode

from visualization.util import render_graph


def get_node_color(model: LockingNode):
    """
    Give each of the locking nodes a proper recognizable color.
    """
    if model.node_type == model.node_type.ENTRY:
        return "#488AC7"
    elif model.node_type == model.node_type.SUCCESS:
        return "#6AA121"
    elif model.node_type == model.node_type.FAILURE:
        return "#ED1C24"
    else:
        return "#CCCCCC"


def render_locking_structure(model: AtomicNode):
    """
    Render the graph within the given atomic node.
    """
    # node_labels = {n: str(n.partner) for n in model.graph.nodes}
    node_labels = {
        n: "%s: %s%s" % (
            n.partner,
            "+" if n.node_type.value == 1 else "-",
            n.target_locks if len(n.target_locks) > 0 else "{}"
        ) for n in model.graph.nodes
    }

    render_graph(model.graph, title=str(model.partner), node_color_func=get_node_color, labels=node_labels)
