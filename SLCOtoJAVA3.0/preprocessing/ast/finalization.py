from grouping.resolver import set_groupings
from locking.identities import assign_lock_identities, get_lock_id_requests, generate_lock_data, assign_lock_request_ids
from objects.ast.models import Transition, StateMachine, Class, SlcoModel, Object


def finalize_state_machine(e: StateMachine):
    set_groupings(e)
    for g in e.state_to_decision_node.values():
        generate_lock_data(g)
        e.max_number_of_lock_requests = max(assign_lock_request_ids(g, 0), e.max_number_of_lock_requests)
    pass


def finalize_class(e: Class):
    # Assign lock identities to the model.
    assign_lock_identities(e)

    # Finalize the transitions and state machines once the lock identities have been assigned.
    for sm in e.state_machines:
        finalize_state_machine(sm)


def finalize_model(e: SlcoModel):
    for c in e.classes:
        finalize_class(c)


def finalize(e: SlcoModel):
    finalize_model(e)
