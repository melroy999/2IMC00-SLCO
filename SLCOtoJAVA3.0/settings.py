# Add a limit on the number of iterations.
import os

# The target model and folder.
model_name = ""
model_folder = ""


def init(parameters):
    """Initialize the global variables, defining the settings of the program"""
    global model_folder, model_name

    # Add the name of the model and the location.
    model_folder, model_name = os.path.split(parameters.model)
