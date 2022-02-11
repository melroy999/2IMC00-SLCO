from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from objects.ast.models import Variable


def render_variable_default_value(model: Variable) -> str:
    """Render the default value of the given variable."""
    a = model.def_values if model.is_array else model.def_value
    type_name = "boolean" if model.is_boolean else "char" if model.is_byte else "int"
    if model.is_array:
        default_value = f"new {type_name}[] {{ {', '.join(map(str, a)).lower()} }}"
    elif model.is_byte:
        default_value = f"(char) {a}"
    else:
        default_value = a
    return default_value
