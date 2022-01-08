import logging
from typing import Union

from objects.ast.models import StateMachine, DecisionNode, Transition, State
from smt.optimization import create_decision_groupings


def set_groupings(model: StateMachine):
    """
    Assign groupings of transitions to the given state machine.
    """
    logging.info(f"> Assigning decision groupings for state machine {model}")
    # Create a mapping between starting state and the transitions starting therein.
    for t in model.transitions:
        model.state_to_transitions[t.source].append(t)

    # Create a decision structure.
    for state, transitions in model.state_to_transitions.items():
        logging.info(f"Assigning decision groupings for state {state}")
        root_decision_node = model.state_to_decision_node[state] = create_decision_groupings(
            transitions
        )
        print_decision_structure(root_decision_node)


def print_decision_structure(model: Union[Transition, DecisionNode], indents=0) -> None:
    """
    Write the decision structure to the log in a human readable format.
    """
    if isinstance(model, DecisionNode):
        if model.is_deterministic:
            logging.info(f"{' ' * indents} - DET:")
            for decision in model.decisions:
                print_decision_structure(decision, indents + 2)
        else:
            logging.info(f"{' ' * indents} - N_DET:")
            for decision in model.decisions:
                print_decision_structure(decision, indents + 2)
    else:
        logging.info(
            f"{' ' * indents} - p:{model.priority}, id:{model.id} {model.source} -> {model.target}: {model.guard}"
        )
