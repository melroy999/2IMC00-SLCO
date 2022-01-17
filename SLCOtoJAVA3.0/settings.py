# Add a limit on the number of iterations.
import os

# The target model and folder.
model_name = ""
model_folder = ""

# Should the locking system render statements that verify whether target variables have been locked before use?
verify_locks = False

# What decision structures should be used? Sequential of False, Random pick if True.
non_determinism = False

# Should the sequential decision grouping be considered an atomic operation?
atomic_sequential = False

# Lock the entirety of the array instead of single elements.
lock_full_arrays = False

# Make the execution of the program sequential by using the same lock for all of the statements using class variables.
statement_locks = False

# Visualize the locking graph.
visualize_locking_graph = False


def init(parameters):
    """Initialize the global variables, defining the settings of the program"""
    global model_folder, model_name, verify_locks, non_determinism, atomic_sequential, visualize_locking_graph, \
        lock_full_arrays, statement_locks

    # Add the name of the model and the location.
    model_folder, model_name = os.path.split(parameters.model)

    # Visualize data.
    visualize_locking_graph = parameters.visualize_locking_graph

    # Store all of the desired settings.
    verify_locks = parameters.verify_locks
    non_determinism = parameters.non_determinism
    atomic_sequential = parameters.atomic_sequential
    lock_full_arrays = parameters.lock_full_arrays
    statement_locks = parameters.statement_locks
