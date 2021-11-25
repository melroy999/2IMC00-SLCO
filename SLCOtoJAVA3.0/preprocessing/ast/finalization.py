from grouping.resolver import set_groupings
from locking.identities import assign_lock_identities
from locking.structuring import create_locking_structure
from objects.ast.models import Transition, StateMachine, Class, SlcoModel, Object


def finalize_state_machine(e: StateMachine):
    set_groupings(e)
    # for g in e.state_to_decision_node.values():
    #     generate_lock_data(g)
    #     move_lock_acquisition_data(g)
    #     generate_locking_phases(g)
    #     e.max_number_of_lock_requests = max(assign_lock_request_ids(g, 0), e.max_number_of_lock_requests)

    for t in e.transitions:
        create_locking_structure(t)


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
