# Import the necessary libraries.
import argparse
import logging
import os
import pathlib
import sys

import settings
from overrides.slcolibrev import read_SLCO_model
from objects.ast.util import ast_to_model
from preprocessing.simplification import simplify_slco_model
from preprocessing.ast.restructuring import restructure
from preprocessing.ast.simplification import simplify
from preprocessing.ast.finalization import finalize
from rendering.java.renderer import JavaModelRenderer
from rendering.measurements.counting.renderer import CountMeasurementsModelRenderer
from rendering.measurements.logging.renderer import LogMeasurementsModelRenderer
from rendering.vercors.locking.renderer import VercorsLockingStructureModelRenderer, \
    VercorsLockingCoverageModelRenderer, VercorsLockingRewriteRulesModelRenderer
from rendering.vercors.structure.renderer import VercorsStructureModelRenderer


def preprocess(model):
    """Gather additional data about the model."""
    logging.info(f">>> Preprocessing model \"{model}\"")
    logging.info(f">> Converting \"{model}\" to the code generator model")
    model = ast_to_model(model, dict())

    # Restructure the model.
    logging.info(f">> Restructuring model \"{model}\"")
    restructure(model)
    logging.info(f">> Simplifying model \"{model}\"")
    simplify(model)
    logging.info(f">> Finalizing model \"{model}\"")
    finalize(model)

    return model


def render(model, model_folder):
    """The translation function."""
    # Create the result folder.
    pathlib.Path(os.path.join(model_folder, "results")).mkdir(parents=True, exist_ok=True)

    # Write the program to the desired output file.
    file_name = os.path.join(model_folder, "results", model.name + ".java")
    logging.info(f">>> Rendering model \"{model}\" to file \"{file_name}\"")
    with open(file_name, "w") as out_file:
        out_file.write(JavaModelRenderer().render_model(model))

    if settings.performance_measurements:
        # Create the logging and counting folders.
        pathlib.Path(os.path.join(model_folder, "results", "logging")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(model_folder, "results", "counting")).mkdir(parents=True, exist_ok=True)

        # Write the program to the desired output file.
        file_name = os.path.join(model_folder, "results", "logging", model.name + ".java")
        logging.info(f">>> Rendering measurement model \"{model}\" to file \"{file_name}\"")
        with open(file_name, "w") as out_file:
            out_file.write(LogMeasurementsModelRenderer().render_model(model))

        # Write the program to the desired output file.
        file_name = os.path.join(model_folder, "results", "counting", model.name + ".java")
        logging.info(f">>> Rendering measurement model \"{model}\" to file \"{file_name}\"")
        with open(file_name, "w") as out_file:
            out_file.write(CountMeasurementsModelRenderer().render_model(model))

    if settings.vercors_verification:
        # Create the structure and locking folders.
        pathlib.Path(os.path.join(model_folder, "results", "vercors", "structure")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(model_folder, "results", "vercors", "locking")).mkdir(parents=True, exist_ok=True)

        # Write the program to the desired output file.
        file_name = os.path.join(model_folder, "results", "vercors", "structure", model.name + ".java")
        logging.info(f">>> Rendering vercors model \"{model}\" to file \"{file_name}\"")
        with open(file_name, "w") as out_file:
            out_file.write(VercorsStructureModelRenderer().render_model(model))

        # Write the program to the desired output file.
        file_name = os.path.join(model_folder, "results", "vercors", "locking", model.name + "_p1_structure.java")
        logging.info(f">>> Rendering vercors model \"{model}\" to file \"{file_name}\"")
        with open(file_name, "w") as out_file:
            out_file.write(VercorsLockingStructureModelRenderer().render_model(model))

        # Write the program to the desired output file.
        file_name = os.path.join(model_folder, "results", "vercors", "locking", model.name + "_p2_coverage.java")
        logging.info(f">>> Rendering vercors model \"{model}\" to file \"{file_name}\"")
        with open(file_name, "w") as out_file:
            out_file.write(VercorsLockingCoverageModelRenderer().render_model(model))

        # Write the program to the desired output file.
        file_name = os.path.join(model_folder, "results", "vercors", "locking", model.name + "_p3_rewrite_rules.java")
        logging.info(f">>> Rendering vercors model \"{model}\" to file \"{file_name}\"")
        with open(file_name, "w") as out_file:
            out_file.write(VercorsLockingRewriteRulesModelRenderer().render_model(model))


def get_argument_parser():
    """Get a parser for the input arguments."""
    parser = argparse.ArgumentParser(
        description="Transform an SLCO 2.0 model to a Java program", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model", help="The SLCO 2.0 model to be converted to a Java program.")

    # Parameters that control the locking structure.
    parser.add_argument("-use_random_pick", action="store_true", help="Use non-deterministic structures instead of "
                                                                      "relying on the priority and list ordering.")
    parser.add_argument("-no_deterministic_structures", action="store_true", help="Disable the creation of "
                                                                                  "deterministic structures and force "
                                                                                  "the decision structure to choose a "
                                                                                  "transition arbitrarily.")

    parser.add_argument("-decision_structure_solver_id", nargs="?", type=int, choices=range(0, 6), const=1, default=0,
                        required=False, help=
                        "The ID of the decision structure solver to use:\n"
                        "0. Greedy basic solver meeting minimum requirements\n"
                        "1. Greedy solver that merges equal transitions (default)\n"
                        "2. Greedy solver that creates a nested structure for contained transitions\n"
                        "3. Optimal basic solver meeting minimum requirements\n"
                        "4. Optimal solver that merges equal transitions\n"
                        "5. Optimal solver that creates a nested structure for contained transitions\n"
                        )

    # Parameters that control the locking mechanism.
    parser.add_argument("-visualize_locking_graph", action="store_true", help="Create a graph visualization of the "
                                                                              "locking graph.")
    parser.add_argument("-no_locks", action="store_true", help="Create faulty code that does not perform locking and "
                                                               "hence will not meet the requirement of atomicity.")
    parser.add_argument("-statement_level_locking", action="store_true", help="Perform locking at the statement level.")
    parser.add_argument("-lock_array", action="store_true", help="Lock the array instead of an individual index.")

    # Parameters that control which statements are rendered.
    parser.add_argument("-verify_locks", action="store_true", help="Add Java statements that verify whether locks have "
                                                                   "been acquired before use.")
    parser.add_argument("-iteration_limit", nargs="?", type=int, const=10000, default=0, required=False,
                        help="Produce a transition counter in the code, to make program executions finite "
                             "(default: 10000 iterations).")
    parser.add_argument("-running_time", nargs="?", type=int, const=60, default=0, required=False,
                        help="Add a timer to the code, to make program executions finite (in seconds, default: 60s).")

    # Parameters that control the performance measurements code.
    parser.add_argument("-log_file_size", nargs="?", type=str, default="100MB", required=False,
                        help="The rollover size for log files generated "
                             "during logging driven performance measurements.")
    parser.add_argument("-log_buffer_size", nargs="?", type=int, default=4194304, required=False,
                        help="The buffer size for the logger used in logging driven performance measurements.")
    parser.add_argument("-compression_level", nargs="?", type=int, const=3, default=3, required=False,
                        help="The buffer size for the logger used in logging driven performance measurements.")
    parser.add_argument("-package_name", nargs="?", type=str, default="", required=False,
                        help="The name of the root package the model should be part of.")

    # Control which models are rendered.
    parser.add_argument("-vercors_verification", action="store_true", help="Render models that uses VerCors to "
                                                                           "formally verify the generated model.")
    parser.add_argument("-performance_measurements", action="store_true", help="Render models that uses measure the "
                                                                               "performance of the generated model.")

    return parser


def report_parsing_errors(parser: argparse.ArgumentParser, parsed_arguments):
    """Report any errors found in the parsed values."""
    if parsed_arguments.iteration_limit != 0 and parsed_arguments.running_time != 0:
        parser.error("The arguments -iteration_limit and -running_time are mutually exclusive.")


def main(_args):
    """The main function."""
    # First, set up the logging format.
    level = logging.INFO
    fmt = "[%(levelname)s][%(asctime)s][%(module)s]: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logging.info("#" * 180)
    logging.info(f"Starting the Java code generation component with the arguments {_args}")

    # Define the arguments that the program supports and parse accordingly.
    parser = get_argument_parser()
    parsed_arguments = parser.parse_args(_args)
    report_parsing_errors(parser, parsed_arguments)

    logging.info(f"Parsed arguments: {parsed_arguments}")
    logging.info("#" * 180)

    # Parse the parameters and save the settings.
    settings.init(parsed_arguments, _args)

    # Read the model.
    model_path = os.path.join(settings.model_folder, settings.model_name)
    model = read_SLCO_model(model_path)
    model = simplify_slco_model(model)

    # Preprocess the model.
    model = preprocess(model)

    # Render the model.
    render(model, settings.model_folder)


if __name__ == "__main__":
    args = []
    for i in range(1, len(sys.argv)):
        args.append(sys.argv[i])
    main(args)
