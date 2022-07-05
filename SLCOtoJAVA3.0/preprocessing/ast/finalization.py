import logging

from grouping.resolver import set_groupings
from locking.identities import assign_lock_identities
from locking.structuring import initialize_main_locking_structure, \
    finalize_locking_structure
from objects.ast.models import Class, SlcoModel


def finalize_class(e: Class):
    """Finalize the given SLCO class object."""
    logging.info(f">> Finalizing {e}.")

    # Finalize the transitions and state machines.
    for sm in e.state_machines:
        logging.info(f"> Assigning decision groupings for {sm}.")
        set_groupings(sm)

    # Assign a locking structure to all containing objects.
    logging.info(f"> Initializing the main locking structure of {e}.")
    for sm in e.state_machines:
        for s in sm.states:
            initialize_main_locking_structure(sm, s)

    # Assign lock identities to the model.
    logging.info(f"> Assigning lock identities to {e}.")
    assign_lock_identities(e)

    # Finalize the locking structure of all containing objects.
    logging.info(f"> Finalizing the locking structure of {e}.")
    for sm in e.state_machines:
        for s in sm.states:
            finalize_locking_structure(sm, s)


def finalize_model(e: SlcoModel):
    """Finalize the given SLCO model object."""
    for c in e.classes:
        finalize_class(c)


def finalize(e: SlcoModel):
    """Finalize the given model."""
    logging.info(f"> Finalizing {e}.")
    finalize_model(e)
