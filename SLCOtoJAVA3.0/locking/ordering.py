# TODO: Create a routine that maps the internal ordering of variables in top-level expressions and primaries
from typing import Iterable

import objects.ast.models

# Type abbreviations
SLCOModel = objects.ast.models.SLCOModel
Primary = objects.ast.models.Primary
Expression = objects.ast.models.Expression


def get_top_level_evaluables(model: Iterable):
    if isinstance(model, (Primary, Expression)):
        # TODO: The ordering only matters for and operations.
        pass
    else:
        for o in model:
            get_top_level_evaluables(o)
