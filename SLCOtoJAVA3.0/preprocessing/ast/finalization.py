from locking.identities import assign_lock_identities, get_lock_id_requests
from objects.ast.models import Transition, StateMachine, Class
from objects.ast.util import get_weighted_variable_dependency_graph, get_weighted_class_variable_dependency_graph


def finalize_transition(e: Transition):
    # Create a list of targets that need to be requested by the locking mechanism.
    for s in e.statements:
        s.lock_id_requests = get_lock_id_requests(s)


def finalize_state_machine(e: StateMachine):
    for t in e.transitions:
        finalize_transition(t)
        e.state_to_transitions[t.source].append(t)


def finalize_class(e: Class):
    # Create a weighted dependency graph such that lock identities can be assigned.
    e.weighted_variable_dependency_graph = get_weighted_variable_dependency_graph(e)
    e.weighted_class_variable_dependency_graph = get_weighted_class_variable_dependency_graph(e)

    # Assign lock identities to the model.
    assign_lock_identities(e)

    # Finalize the transitions and state machines once the lock identities have been assigned.
    for sm in e.state_machines:
        finalize_state_machine(sm)
