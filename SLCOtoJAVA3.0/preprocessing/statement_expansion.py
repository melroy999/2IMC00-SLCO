from __future__ import annotations
from typing import Union

import objects.ast.models as models


def expand_result(func):
    """Create a decorator that automatically beautifies the return value of a function."""
    def function_wrapper(e):
        result = func(e)
        print(e)
        return result
    return function_wrapper


def expand_expression(e: models.Expression):
    pass



def expand(e: Union[models.Expression, models.Primary, models.VariableRef, models.Composite, models.Assignment]):
    """Expand the given object"""
    if isinstance(e, models.Expression):
        pass

    return e