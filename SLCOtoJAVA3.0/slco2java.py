# Import the necessary libraries.
import argparse
import logging
import os
import sys

import settings
from overrides.slcolibrev import read_SLCO_model
from objects.ast.util import ast_to_model
from preprocessing.simplification import simplify_slco_model
from preprocessing.ast.restructuring import restructure
from preprocessing.ast.simplification import simplify
from preprocessing.ast.finalization import finalize
from rendering.java.renderer import JavaModelRenderer
from rendering.measurements.renderer import LogMeasurementsModelRenderer
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
    # Write the program to the desired output file.
    file_name = os.path.join(model_folder, model.name + ".java")
    logging.info(f">>> Rendering model \"{model}\" to file \"{file_name}\"")
    with open(file_name, 'w') as out_file:
        out_file.write(JavaModelRenderer().render_model(model))

    # Write the program to the desired output file.
    file_name = os.path.join(model_folder, model.name + "_vercors_structure.java")
    logging.info(f">>> Rendering vercors model \"{model}\" to file \"{file_name}\"")
    with open(file_name, 'w') as out_file:
        out_file.write(VercorsStructureModelRenderer().render_model(model))

    # Write the program to the desired output file.
    file_name = os.path.join(model_folder, model.name + "_vercors_locking_p1_structure.java")
    logging.info(f">>> Rendering vercors model \"{model}\" to file \"{file_name}\"")
    with open(file_name, 'w') as out_file:
        out_file.write(VercorsLockingStructureModelRenderer().render_model(model))

    # Write the program to the desired output file.
    file_name = os.path.join(model_folder, model.name + "_vercors_locking_p2_coverage.java")
    logging.info(f">>> Rendering vercors model \"{model}\" to file \"{file_name}\"")
    with open(file_name, 'w') as out_file:
        out_file.write(VercorsLockingCoverageModelRenderer().render_model(model))

    # Write the program to the desired output file.
    file_name = os.path.join(model_folder, model.name + "_vercors_locking_p3_rewrite_rules.java")
    logging.info(f">>> Rendering vercors model \"{model}\" to file \"{file_name}\"")
    with open(file_name, 'w') as out_file:
        out_file.write(VercorsLockingRewriteRulesModelRenderer().render_model(model))

    # Write the program to the desired output file.
    file_name = os.path.join(model_folder, model.name + "_measurements.java")
    logging.info(f">>> Rendering vercors model \"{model}\" to file \"{file_name}\"")
    with open(file_name, 'w') as out_file:
        out_file.write(LogMeasurementsModelRenderer().render_model(model))


def get_argument_parser():
    """Get a parser for the input arguments."""
    parser = argparse.ArgumentParser(description="Transform an SLCO 2.0 model to a Java program")
    parser.add_argument("model", help="The SLCO 2.0 model to be converted to a Java program.")

    # Parameters that control the locking structure.
    parser.add_argument("-use_random_pick", action='store_true', help="Use non-deterministic structures instead of "
                                                                      "relying on the priority and list ordering.")
    parser.add_argument("-no_deterministic_structures", action='store_true', help="Disable the creation of "
                                                                                  "deterministic structures and force "
                                                                                  "the decision structure to choose a "
                                                                                  "transition arbitrarily.")
    parser.add_argument("-use_full_smt_dsc", action='store_true', help="(EXPERIMENTAL) Use the alternative algorithm "
                                                                       "for constructing decision structures that uses "
                                                                       "only a single SMT model to assign all "
                                                                       "transitions to groups.")

    # Parameters that control the locking mechanism.
    parser.add_argument("-atomic_sequential", action='store_true', help="Make the sequential decision structures an "
                                                                        "atomic operation.")
    parser.add_argument("-lock_full_arrays", action='store_true', help="Lock the entirety of an array instead of a "
                                                                       "single element.")
    parser.add_argument("-statement_locks", action='store_true', help="Make the execution sequential by having each "
                                                                      "statement use the same lock.")
    parser.add_argument("-visualize_locking_graph", action='store_true', help="Create a graph visualization of the "
                                                                              "locking graph.")

    # Parameters that control which statements are rendered.
    parser.add_argument("-verify_locks", action='store_true', help="Add Java statements that verify whether locks have "
                                                                   "been acquired before use.")
    parser.add_argument("-iteration_limit", nargs="?", type=int, const=10000, default=0, required=False,
                        help="Produce a transition counter in the code, to make program executions finite "
                             "(default: 10000 iterations).")
    parser.add_argument("-running_time", nargs="?", type=int, const=60, default=0, required=False,
                        help="Add a timer to the code, to make program executions finite (in seconds, default: 60s).")
    return parser


def report_parsing_errors(parser: argparse.ArgumentParser, parsed_arguments):
    """Report any errors found in the parsed values."""
    if parsed_arguments.atomic_sequential and parsed_arguments.use_random_pick:
        parser.error("The arguments -atomic_sequential and -use_random_pick are mutually exclusive.")
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
    settings.init(parsed_arguments)

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
