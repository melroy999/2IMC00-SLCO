from objects.ast.models import StateMachine, DecisionNode, GuardNode


def set_groupings(model: StateMachine):
    """
    Assign groupings of transitions to the given state machine.
    """
    # Create a mapping between starting state and the transitions starting therein.
    for t in model.transitions:
        model.state_to_transitions[t.source].append(t)

    # Give all of the transitions an unique ID per source grouping.
    for transitions in model.state_to_transitions.values():
        for i, t in enumerate(transitions):
            t.id = i

    # Create a decision structure.
    for state, transitions in model.state_to_transitions.items():
        # TEMP: Wrap all transitions in a deterministic decision block.
        decisions = [GuardNode(t) for t in transitions]
        model.state_to_decision_node[state] = DecisionNode(decisions, True)
