from typing import Union

from objects.ast.models import Primary, Expression
from objects.ast.util import get_variable_references, get_class_variable_references


def get_order_violations():
    pass


def get_bound_check_violations(model: Union[Primary, Expression]):
    """
    Find possible bound checks in the model pertaining to class variable array accesses.
    """
    # Bound check violations can only occur between clauses in logic gates (and, or, xor).
    if isinstance(model, Primary):
        if model.body is not None:
            get_bound_check_violations(model.body)
    elif isinstance(model, Expression) and model.op in ["and", "or", "xor"]:
        # Find the variables used in each of the clauses.
        variable_references = [get_variable_references(e) for e in model.values]

        for i, e in enumerate(model.values[1:]):
            # Find the variables used within the global array variables of the given expression.
            class_array_variable_references = [
                r for r in variable_references[i + 1] if r.var.is_array and r.var.is_class_variable
            ]

            pass
