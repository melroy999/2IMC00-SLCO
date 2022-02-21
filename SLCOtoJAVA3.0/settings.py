# Add a limit on the number of iterations.
import logging
import os

# The target model and folder.
model_name = ""
model_folder = ""

# What decision structures should be used? Sequential of False, Random pick if True.
use_random_pick = False

# Should the program generate Java code without deterministic structures?
no_deterministic_structures = False

# Should the sequential decision grouping be considered an atomic operation?
atomic_sequential = False

# Lock the entirety of the array instead of single elements.
lock_full_arrays = False

# Make the execution of the program sequential by using the same lock for all of the statements using class variables.
statement_locks = False

# Visualize the locking graph.
visualize_locking_graph = False

# Should the locking system render statements that verify whether target variables have been locked before use?
verify_locks = False

# Set the number of iterations that state machines within the model make.
iteration_limit = 0

# Set the number of seconds the state machines within the model run.
running_time = 0


def init(parameters):
    """Initialize the global variables, defining the settings of the program"""
    global model_folder, model_name, use_random_pick, no_deterministic_structures, atomic_sequential, lock_full_arrays,\
        statement_locks, visualize_locking_graph, verify_locks, iteration_limit, running_time

    # Add the name of the model and the location.
    model_folder, model_name = os.path.split(parameters.model)

    # Store all of the desired settings.
    use_random_pick = parameters.use_random_pick
    no_deterministic_structures = parameters.no_deterministic_structures

    atomic_sequential = parameters.atomic_sequential
    lock_full_arrays = parameters.lock_full_arrays
    statement_locks = parameters.statement_locks
    visualize_locking_graph = parameters.visualize_locking_graph

    verify_locks = parameters.verify_locks
    iteration_limit = parameters.iteration_limit
    running_time = parameters.running_time
