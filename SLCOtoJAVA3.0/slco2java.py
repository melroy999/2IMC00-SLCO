# Import the necessary libraries.
import argparse
import os
import sys

import settings
from overrides.slcolibrev import read_SLCO_model
from objects.ast.models import SlcoModel, Transition, StateMachine, Class
from objects.ast.util import ast_to_model, __dfs__
from objects.ast.visualization import visualize_dependency_graph, visualize_weighted_variable_dependency_graph, \
    visualize_expression, visualize_variable_ordering_graph
from preprocessing.ast.finalization import finalize_class
from preprocessing.ast.restructuring import restructure
from preprocessing.ast.simplification import simplify


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

# TODO:
#   - Unused locks due to unpacking can be released rather elegantly:
#   - Suppose that we have X[0..3] and Y[0..2], locking X[Y[i]] with X.id < Y.id
#   - Variables to lock: [X[Y[i]], X[3], Y[i]]
#   - Lock with unpacking: [X[0], X[1], X[2], X[3], Y[i]]
#   - Lock unpack targets: [X[Y[i]]]
#   - Unlock unused variables added by the unpacking: [X[0], X[1], X[2]]
#   - Resulting locks: [X[3], X[Y[i]], Y[i]]

# TODO:
#   - Ensure that i >= 0 and i < 10 and X[i] doesn't turn into an out of bound exception.
#   - Break the conjunction into multiple parts, such that the bound check will fail before requesting X[i].
#   - Split if a proceeding part of the conjunction contains a variable that is used within an array index of one or
#   more variables within the current block.
#   - How to handle nested array indices...? Take the whole array or with the index...?
#   - The latter might result in errors when overly creative...?
#   - Note that this has to work in nested parts of the expression too: (i >= 0 and i < 10 and X[i]) or X[0].
#   - Simplification: top level disjunctions should be handled as separate options instead.
#   - How to avoid xor and equality...?


def preprocess(model):
    """"Gather additional data about the model"""
    model = ast_to_model(model, dict())

    # Restructure the model.
    for t in __dfs__(model, _filter=lambda x: isinstance(x, Transition)):
        restructure(t)
        simplify(t)

    for c in __dfs__(model, _filter=lambda x: isinstance(x, Class)):
        finalize_class(c)

    target_objects = __dfs__(
        model, self_first=True, _filter=lambda x: type(x) in [Transition, SlcoModel, Class, StateMachine]
    )

    for o in target_objects:
        print(o)

        if isinstance(o, Transition):
            for v in o.statements:
                visualize_expression(v)
                visualize_dependency_graph(v)
                visualize_variable_ordering_graph(v)

        if isinstance(o, Class):
            visualize_weighted_variable_dependency_graph(o)


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
