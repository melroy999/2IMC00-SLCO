import logging
from typing import Union, List

import settings
from objects.ast.models import StateMachine, DecisionNode, Transition
from smt.optimization import create_decision_groupings
from smt.solutions.combinations import *


def set_groupings(model: StateMachine):
    """Assign groupings of transitions to the given state machine."""
    # Create a mapping between starting state and the transitions starting therein.
    for t in model.transitions:
        model.state_to_transitions[t.source].append(t)

    # Create a decision structure.
    for state, transitions in model.state_to_transitions.items():
        logging.info(f"Assigning decision groupings for state {state}")
        root_decision_node = model.state_to_decision_node[state] = get_groupings(transitions)
        print_decision_structure(root_decision_node)


def get_groupings(transitions: List[Transition]):
    """Get the groupings for the given list of transitions."""
    solver_options = {
        0: GreedyBaseDecisionStructureSolver,
        1: GreedyEqualsNestedDecisionStructureSolver,
        2: GreedyContainsNestedDecisionStructureSolver,
        3: OptimalBaseDecisionStructureSolver,
        4: OptimalEqualsNestedDecisionStructureSolver,
        5: OptimalContainsNestedDecisionStructureSolver,
    }
    return solver_options[settings.decision_structure_solver_id](transitions).solve()


def print_decision_structure(model: Union[Transition, DecisionNode], indents=0) -> None:
    """Write the decision structure to the log in a human readable format."""
    if isinstance(model, DecisionNode):
        if model.is_deterministic:
            logging.info(f"{' ' * indents} - DET:")
            for decision in model.decisions + model.excluded_transitions:
                print_decision_structure(decision, indents + 2)
        else:
            if model.guard_statement is not None:
                logging.info(f"{' ' * indents} - N_DET ({model.guard_statement.guard}):")
            else:
                logging.info(f"{' ' * indents} - N_DET:")
            for decision in model.decisions + model.excluded_transitions:
                print_decision_structure(decision, indents + 2)
    else:
        logging.info(
            f"{' ' * indents} - {model}"
        )
