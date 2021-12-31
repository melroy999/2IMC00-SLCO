from objects.ast.models import StateMachine, DecisionNode, GuardNode
from smt.optimization import create_minimal_groupings_of_non_overlapping_active_regions


def set_groupings(model: StateMachine):
    """
    Assign groupings of transitions to the given state machine.
    """
    # Create a mapping between starting state and the transitions starting therein.
    for t in model.transitions:
        model.state_to_transitions[t.source].append(t)

    # Create a decision structure.
    for state, transitions in model.state_to_transitions.items():
        model.state_to_decision_node[state] = create_minimal_groupings_of_non_overlapping_active_regions(
            transitions
        )
