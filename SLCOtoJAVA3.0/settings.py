# Add a limit on the number of iterations.
import os

# The target model and folder.
model_name = ""
model_folder = ""

# Parameters that influence the locking behavior.
release_locks_asap = False
release_conflict_resolution_locks = False
priority_queue_locking = False
preserve_lock_list_ordering = False


def init(parameters):
    """Initialize the global variables, defining the settings of the program"""
    global model_folder, model_name, release_locks_asap, release_conflict_resolution_locks, priority_queue_locking, preserve_lock_list_ordering

    # Add the name of the model and the location.
    model_folder, model_name = os.path.split(parameters.model)

    # Make Reentrant locks fair/unfair.
    release_locks_asap = parameters.release_locks_asap
    release_conflict_resolution_locks = parameters.release_conflict_resolution_locks
    priority_queue_locking = parameters.release_locks_asap or parameters.priority_queue_locking
    preserve_lock_list_ordering = not priority_queue_locking and (
            parameters.release_conflict_resolution_locks or parameters.release_locks_asap
    )
