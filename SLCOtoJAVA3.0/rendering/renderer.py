# Initialize the template engine.
import uuid
from typing import Union

import jinja2 as jinja2
import random

from objects.ast.interfaces import SlcoStatementNode
from objects.ast.models import SlcoModel, Object, Class, StateMachine, Transition, Variable, Expression, Assignment, \
    Composite, Primary, VariableRef, DecisionNode, GuardNode

# SUPPORT VARIABLES
r = random.Random()
r.seed(0)
reproducible_seed = uuid.UUID(int=r.getrandbits(128), version=4)


# UTIL FUNCTIONS
def render_type(model: Variable):
    """Render the type of the given variable object."""
    if model.is_array:
        return "%s[]" % ("boolean" if model.is_boolean else "int")
    else:
        return "%s" % ("boolean" if model.is_boolean else "int")


def is_decision_node(model):
    """Return true when the given model is a decision node, false otherwise."""
    return isinstance(model, DecisionNode)


def is_transition(model):
    """Return true when the given model is a transition, false otherwise."""
    return isinstance(model, Transition)


# Conversion from SLCO to Java operators.
java_operator_mappings = {
    "<>": "!=",
    "=": "==",
    "and": "&&",
    "or": "||",
    "not": "!"
}


def render_java_instruction(model: Union[Expression, Primary, VariableRef], rewrite_table=None):
    """Render the given object as a Java instruction."""
    if rewrite_table is None:
        rewrite_table = {}

    if isinstance(model, Expression):
        if model.op == "**":
            left_str = render_java_instruction(model.values[0], rewrite_table)
            right_str = render_java_instruction(model.values[1], rewrite_table)
            return "(int) Math.pow(%s, %s)" % (left_str, right_str)
        elif model.op == "%":
            # The % operator in Java is the remainder operator, which is not the modulo operator.
            left_str = render_java_instruction(model.values[0], rewrite_table)
            right_str = render_java_instruction(model.values[1], rewrite_table)
            return "Math.floorMod(%s, %s)" % (left_str, right_str)
        else:
            values_str = [render_java_instruction(v, rewrite_table) for v in model.values]
            return (" %s " % java_operator_mappings.get(model.op, model.op)).join(values_str)
    elif isinstance(model, Primary):
        if model.value is not None:
            exp_str = str(model.value).lower()
        elif model.ref is not None:
            exp_str = render_java_instruction(model.ref, rewrite_table)
        else:
            exp_str = "(%s)" % render_java_instruction(model.body, rewrite_table)
        return ("!(%s)" if model.sign == "not" else model.sign + "%s") % exp_str
    elif isinstance(model, VariableRef):
        var_str = model.var.name
        if model.index is not None:
            var_str += "[%s]" % render_java_instruction(model.index, rewrite_table)
        return rewrite_table.get(var_str, var_str)
    elif isinstance(model, Transition):
        return "execute_transition_%s_%s()" % (model.source, model.id)
    else:
        raise Exception("This functionality has not yet been implemented.")


# MODEL RENDERING FUNCTIONS
# MODEL
def render_model(model: SlcoModel):
    """Render the SLCO model as Java code."""
    return java_model_template.render(model=model)


def render_lock_manager(_):
    """Render the lock manager of the model."""
    return java_lock_manager.render()


def render_object_instantiation(model: Object):
    """Render the instantiation of the given object."""
    arguments = []
    for i, a in enumerate(model.initial_values):
        v = model.type.variables[i]
        if v.is_array:
            arguments.append("new %s[]{%s}" % ("boolean" if v.is_boolean else "int", ", ".join(map(str, a))))
        else:
            arguments.append(a)

    return java_object_instantiation.render(
        name=model.type.name,
        arguments=arguments
    )


# CLASS
def render_class(model: Class):
    """Render the SLCO class as Java code."""
    return java_class_template.render(model=model)


# STATE MACHINE
def render_state_machine(model: StateMachine):
    """Render the SLCO state machine as Java code."""
    return java_state_machine_template.render(model=model)


# TRANSITION
def render_transition(model: Transition):
    """Render the SLCO state machine as Java code."""
    return java_transition_template.render(model=model)


# DECISION NODES
def render_control_flow_node(model: Union[DecisionNode, GuardNode]):
    """Render a decision node fitting the desired parameters."""
    if isinstance(model, GuardNode):
        return java_guard_template.render(model=model)
    else:
        if model.is_deterministic:
            return java_deterministic_decision_template.render(model=model)
        else:
            return java_non_deterministic_decision_template.render(model=model)


# STATEMENTS
def render_statement(model: SlcoStatementNode):
    """Render the SLCO statement as Java code"""
    if isinstance(model, Expression) or isinstance(model, Primary):
        return java_expression_template.render(model=model)
    elif isinstance(model, Assignment):
        return java_assignment_template.render(model=model)
    elif isinstance(model, Composite):
        return java_composite_template.render(model=model)
    else:
        raise Exception("This functionality has not yet been implemented.")


env = jinja2.Environment(
    loader=jinja2.FileSystemLoader("jinja2_templates"),
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=["jinja2.ext.loopcontrols", "jinja2.ext.do"]
)

# Register the rendering filters
env.filters["render_class"] = render_class
env.filters["render_lock_manager"] = render_lock_manager
env.filters["render_state_machine"] = render_state_machine
env.filters["render_transition"] = render_transition
env.filters["render_control_flow_node"] = render_control_flow_node
env.filters["render_statement"] = render_statement

# Register the utility filters
env.filters["render_type"] = render_type
env.filters["render_object_instantiation"] = render_object_instantiation
env.filters["render_java_instruction"] = render_java_instruction
env.filters["is_decision_node"] = is_decision_node
env.filters["is_transition"] = is_transition

# env.filters["get_instruction"] = get_instruction
# env.filters["get_guard_statement"] = get_guard_statement
# env.filters["get_decision_structure"] = get_decision_structure
# env.filters["get_lock_id_list"] = get_lock_id_list

# Load the Java templates.
java_model_template = env.get_template("objects/java_model.jinja2template")
java_class_template = env.get_template("objects/java_class.jinja2template")
java_state_machine_template = env.get_template("objects/java_state_machine.jinja2template")
java_transition_template = env.get_template("objects/java_transition.jinja2template")

java_assignment_template = env.get_template("objects/java_assignment.jinja2template")
java_expression_template = env.get_template("objects/java_expression.jinja2template")
java_composite_template = env.get_template("objects/java_composite.jinja2template")

java_deterministic_decision_template = env.get_template(
    "objects/decision_node/java_deterministic_decision.jinja2template"
)
java_non_deterministic_decision_template = env.get_template(
    "objects/decision_node/java_non_deterministic_decision.jinja2template"
)
java_guard_template = env.get_template(
    "objects/java_guard.jinja2template"
)

java_lock_manager = env.get_template("locking/java_lock_manager.jinja2template")
java_object_instantiation = env.get_template("util/java_object_instantiation.jinja2template")
