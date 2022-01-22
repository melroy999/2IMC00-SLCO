from grouping.resolver import set_groupings
from locking.identities import assign_lock_identities
from locking.structuring import get_locking_structure, finalize_locking_structure
from objects.ast.models import Class, SlcoModel


def finalize_class(e: Class):
    # TODO Start by initializing the locking structures of the individual transitions.
    for sm in e.state_machines:
        for t in sm.transitions:
            pass
        
    # Finalize the transitions and state machines.
    for sm in e.state_machines:
        set_groupings(sm)

    # Assign a locking structure to all containing objects.
    for sm in e.state_machines:
        for s in sm.states:
            get_locking_structure(sm, s)

    # Assign lock identities to the model.
    assign_lock_identities(e)

    # Finalize the locking structure of all containing objects.
    for sm in e.state_machines:
        for s in sm.states:
            finalize_locking_structure(sm, s)


def finalize_model(e: SlcoModel):
    for c in e.classes:
        finalize_class(c)


def finalize(e: SlcoModel):
    finalize_model(e)
