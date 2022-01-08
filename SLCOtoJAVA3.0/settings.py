# Add a limit on the number of iterations.
import os

# The target model and folder.
model_name = ""
model_folder = ""

# Should the locking system render statements that verify whether target variables have been locked before use?
verify_locks = False

# What decision structures should be used?
non_determinism = False

# Should the control flow return to the decision structure if the transition fails outside of the guard statement?
resume_decision_structure = False


def init(parameters):
    """Initialize the global variables, defining the settings of the program"""
    global model_folder, model_name, verify_locks, resume_decision_structure, non_determinism

    # Add the name of the model and the location.
    model_folder, model_name = os.path.split(parameters.model)

    # Store all of the desired settings.
    verify_locks = parameters.verify_locks
    resume_decision_structure = parameters.resume_decision_structure
    non_determinism = parameters.non_determinism
