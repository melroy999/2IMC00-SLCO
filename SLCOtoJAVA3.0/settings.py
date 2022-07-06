# Add a limit on the number of iterations.
import os

# The target model and folder.
model_name = ""
model_folder = ""

# What decision structures should be used? Sequential of False, Random pick if True.
use_random_pick = False

# Should the program generate Java code without deterministic structures?
no_deterministic_structures = False

# The decision structure solver to use.
decision_structure_solver_id = 2

# Do not perform locking.
no_locks = False

# Perform locking at a statement level.
statement_level_locking = False

# Lock the entire array instead of an individual index.
lock_array = False

# Visualize the locking graph.
visualize_locking_graph = False

# Should the locking system render statements that verify whether target variables have been locked before use?
verify_locks = False

# Set the number of iterations that state machines within the model make.
iteration_limit = 0

# Set the number of seconds the state machines within the model run.
running_time = 0

# The log file size used during logging measurements.
log_file_size = "100MB"

# The log buffer size used during logging measurements.
log_buffer_size = 4194304

# The compression level to be used during the logging.
compression_level = 3

# A string to be included during logging.
settings_abbreviations = ""

# The original arguments.
original_arguments = ""

# Indicate that models containing vercors verification code needs to be rendered.
vercors_verification = False

# Indicate that models containing performance measurements needs to be rendered.
performance_measurements = False

# The name of the root package.
package_name = ""


def set_settings_abbreviations(parameters):
    """Initialize the settings abbreviations string"""
    global model_folder, model_name, use_random_pick, no_deterministic_structures, decision_structure_solver_id, \
        atomic_sequential,no_locks, statement_level_locking, lock_array, visualize_locking_graph, verify_locks, \
        iteration_limit, running_time, settings_abbreviations
    included_settings = []
    if use_random_pick != parameters.use_random_pick:
        included_settings.append("URP")
    if no_deterministic_structures != parameters.no_deterministic_structures:
        included_settings.append("NDS")
    if decision_structure_solver_id != parameters.decision_structure_solver_id:
        included_settings.append(f"DSSI={parameters.decision_structure_solver_id}")

    if no_locks != parameters.no_locks:
        included_settings.append("NL")
    if statement_level_locking != parameters.statement_level_locking:
        included_settings.append("SLL")
    if lock_array != parameters.lock_array:
        included_settings.append("LA")

    if verify_locks != parameters.verify_locks:
        included_settings.append("VL")
    if iteration_limit != parameters.iteration_limit:
        included_settings.append(f"I={parameters.iteration_limit}")
    if running_time != parameters.running_time:
        included_settings.append(f"T={parameters.running_time}s")

    # included_settings.append(f"LFS={parameters.log_file_size}")
    # included_settings.append(f"LBS={parameters.log_buffer_size}")
    # included_settings.append(f"CL={parameters.compression_level}")

    included_settings.sort()
    settings_abbreviations = "" if len(included_settings) == 0 else f'[{",".join(included_settings)}]'


def init(parameters, _args):
    """Initialize the global variables, defining the settings of the program"""
    global model_folder, model_name, use_random_pick, no_deterministic_structures, decision_structure_solver_id, \
        no_locks, statement_level_locking, lock_array, visualize_locking_graph, verify_locks, iteration_limit, \
        running_time, log_file_size, log_buffer_size, compression_level, original_arguments, vercors_verification, \
        performance_measurements, package_name

    # Add abbreviations such that used settings can be easily tracked.
    set_settings_abbreviations(parameters)

    # Add the name of the model and the location.
    model_folder, model_name = os.path.split(parameters.model)

    # Store all of the desired settings.
    use_random_pick = parameters.use_random_pick
    no_deterministic_structures = parameters.no_deterministic_structures
    decision_structure_solver_id = parameters.decision_structure_solver_id

    no_locks = parameters.no_locks
    statement_level_locking = parameters.statement_level_locking
    lock_array = parameters.lock_array
    visualize_locking_graph = parameters.visualize_locking_graph

    verify_locks = parameters.verify_locks
    iteration_limit = parameters.iteration_limit
    running_time = parameters.running_time

    log_file_size = parameters.log_file_size
    log_buffer_size = parameters.log_buffer_size
    compression_level = parameters.compression_level

    vercors_verification = parameters.vercors_verification
    performance_measurements = parameters.performance_measurements
    package_name = parameters.package_name

    original_arguments = " ".join(_args)
