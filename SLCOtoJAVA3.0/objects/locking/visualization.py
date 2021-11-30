from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from objects.ast.models import Assignment

if TYPE_CHECKING:
    from objects.locking.models import LockingNode, AtomicNode

from visualization.util import render_graph


def get_node_color(model: LockingNode):
    """
    Give each of the locking nodes a proper recognizable color.
    """
    if model.node_type == model.node_type.ENTRY:
        if isinstance(model.partner, Assignment) or len(model.partner.locking_atomic_node.child_atomic_nodes) == 0:
            return "#488AC7"
        else:
            return "#BCD2E8"
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
    logging.info(f"Visualizing the atomic node graph of object {model.partner}")
    node_labels = dict()
    n: LockingNode
    for n in model.graph.nodes:
        node_labels[n] = ("%s%s%s%s" % (
            n.partner,
            (" +%s" % {str(r) for r in n.locks_to_acquire}).replace("'", "") if len(n.locks_to_acquire) > 0 else "",
            (" -%s" % {str(r) for r in n.locks_to_release}).replace("'", "") if len(n.locks_to_release) > 0 else "",
            (" !%s" % {
                str(r) for r in n.locks_to_acquire if r.is_dirty
            }).replace("'", "") if len(list(r for r in n.locks_to_acquire if r.is_dirty)) > 0 else "",
        )).strip()
        # node_labels[n] = ("%s %s %s" % (
        #     n.partner,
        #     ("+%s" % {r for r in n.locks_to_acquire}).replace("'", "") if len(n.locks_to_acquire) > 0 else "",
        #     ("-%s" % {r for r in n.locks_to_release}).replace("'", "") if len(n.locks_to_release) > 0 else ""
        # )).strip()
    render_graph(model.graph, title=str(model.partner), node_color_func=get_node_color, labels=node_labels)
