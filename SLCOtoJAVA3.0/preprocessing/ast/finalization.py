from grouping.resolver import set_groupings
from locking.identities import assign_lock_identities
from locking.structuring import get_locking_structure, finalize_locking_structure
from objects.ast.models import StateMachine, Class, SlcoModel


def finalize_state_machine(e: StateMachine):
    set_groupings(e)


def finalize_class(e: Class):
    # Finalize the transitions and state machines.
    for sm in e.state_machines:
        finalize_state_machine(sm)

    # Assign a locking structure to all containing objects.
    for sm in e.state_machines:
        for t in sm.transitions:
            get_locking_structure(t)

    # Assign lock identities to the model.
    assign_lock_identities(e)

    # Finalize the locking structure of all containing objects.
    for sm in e.state_machines:
        for t in sm.transitions:
            finalize_locking_structure(t)


def finalize_model(e: SlcoModel):
    for c in e.classes:
        finalize_class(c)


def finalize(e: SlcoModel):
    finalize_model(e)
