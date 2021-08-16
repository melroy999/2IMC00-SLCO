# Import the necessary libraries.
import argparse
import os

import settings
from libraries.slcolib import *
from objects.ast.models import SlcoModel, Transition, StateMachine, Class
from objects.ast.util import copy_node, ast_to_model, __dfs__


# TODO Notes:
#   - the lock ids are to be acquired through a topological ordering--the graph is not a DAG.
#   - [i := 0; x[y[i]] := 1]; due to the assignment replacement, i is replaced by 0.
#       - However, this does mean that i gets locked, ALWAYS. during locking itself, y[0] will be locked instead of y[i]
#       - Hence, it is not an issue--the requirements are never violated, since the full behavior is known.


# TODO:
#   - WARNING!!!!
#   - The idea of nested locking for conjunctions will not work as desired!
#   - By doing so in an if-then-else clause, succeeding decision nodes will never be visited, resulting in a bias!
#   - Hence, only if statement should be used, or a more complex system of methods returning true/false.

# TODO:
#   - WARNING!!!!
#   - The caching in the smt module is sensitive to variables changing type!
#   - Hence, it needs to be reset between state machines.
#   - Or, alternatively, every generated variable needs to be made unique.


def preprocess(model):
    """"Gather additional data about the model"""
    model = ast_to_model(model, dict())
    model2 = copy_node(model, dict())

    target_objects = __dfs__(
        model2, self_first=True, _filter=lambda x: type(x) in [Transition, SlcoModel, Class, StateMachine]
    )

    for o in target_objects:
        print(o)


def get_argument_parser():
    """Get a parser for the input arguments"""
    parser = argparse.ArgumentParser(description="Transform an SLCO 2.0 model to a Java program")
    parser.add_argument("model", help="The SLCO 2.0 model to be converted to a Java program.")
    return parser


def main(_args):
    """The main function"""
    # Define the arguments that the program supports and parse accordingly.
    parser = get_argument_parser()
    parsed_arguments = parser.parse_args(_args)

    # Parse the parameters and save the settings.
    settings.init(parsed_arguments)

    # Read the model.
    model_path = os.path.join(settings.model_folder, settings.model_name)
    model = read_SLCO_model(model_path)

    # Preprocess the model.
    preprocess(model)


if __name__ == '__main__':
    args = []
    for i in range(1, len(sys.argv)):
        args.append(sys.argv[i])
    main(args)
