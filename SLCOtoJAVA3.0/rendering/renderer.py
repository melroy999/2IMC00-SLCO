# Initialize the template engine.
import uuid
from typing import Union, List, Set

import jinja2 as jinja2

import settings
from objects.ast.interfaces import SlcoStatementNode, SlcoLockableNode
from objects.ast.models import SlcoModel, Object, Class, StateMachine, Transition, Variable, Expression, Assignment, \
    Composite, Primary, VariableRef, DecisionNode, GuardNode, LockRequest


# UTIL FUNCTIONS
from rendering.environment_settings import env


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
    # TODO: temporarily disable the renderer.
    return ""
    return java_model_template.render(model=model)


def render_lock_manager(_):
    """Render the lock manager of the model."""
    return java_lock_manager_template.render(
        settings=settings
    )


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
    return java_state_machine_template.render(
        model=model,
        settings=settings
    )


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


# LOCKING
def render_lock_acquisition_block(model: SlcoLockableNode):
    """Render code that acquires the given lock requests."""
    return java_lock_acquisition_block_template.render(
        model=model
    )


def render_lock_acquisition(model: List[LockRequest]):
    """Render code that acquires the given lock requests."""
    return java_lock_acquisition_template.render(
        lock_requests=list(model),
        settings=settings
    )


def render_lock_release_block(model: SlcoLockableNode):
    """Render code that acquires the given lock requests."""
    return java_lock_release_block_template.render(
        model=model
    )


def render_lock_release(model: Set[LockRequest]):
    """Render code that releases the given lock requests."""
    sorted_model = sorted(model, key=lambda r: r.id)

    # Create phases for the unlocking to compensate for possible id gaps.
    lock_release_phases = []
    if not settings.priority_queue_locking:
        current_phase = [sorted_model[0]]
        for lock_request in sorted_model[1:]:
            if lock_request.id - current_phase[-1].id > 1:
                # Flush the current phase and create a new one.
                lock_release_phases.append(current_phase)
                current_phase = [lock_request]
            else:
                current_phase.append(lock_request)
        if len(current_phase) > 0:
            lock_release_phases.append(current_phase)
    else:
        # Phases aren't necessary for priority queue locking, since all of the target locks will be in the queue.
        lock_release_phases.append(sorted_model)

    return java_lock_release_template.render(
        lock_release_phases=lock_release_phases,
        settings=settings
    )


# Register the rendering filters
env.filters["render_class"] = render_class
env.filters["render_lock_manager"] = render_lock_manager
env.filters["render_lock_acquisition_block"] = render_lock_acquisition_block
env.filters["render_lock_acquisition"] = render_lock_acquisition
env.filters["render_lock_release_block"] = render_lock_release_block
env.filters["render_lock_release"] = render_lock_release
env.filters["render_state_machine"] = render_state_machine
env.filters["render_transition"] = render_transition
env.filters["render_control_flow_node"] = render_control_flow_node
env.filters["render_statement"] = render_statement
env.filters["render_type"] = render_type
env.filters["render_object_instantiation"] = render_object_instantiation
env.filters["render_java_instruction"] = render_java_instruction

# Register the utility filters
env.filters["is_decision_node"] = is_decision_node
env.filters["is_transition"] = is_transition

# Load the Java templates.
java_model_template = env.get_template("objects/java_model.jinja2template")
java_class_template = env.get_template("objects/java_class.jinja2template")
java_state_machine_template = env.get_template("objects/java_state_machine.jinja2template")
java_transition_template = env.get_template("objects/java_transition.jinja2template")

java_assignment_template = env.get_template("objects/java_assignment.jinja2template")
java_expression_template = env.get_template("objects/java_expression.jinja2template")
java_composite_template = env.get_template("objects/java_composite.jinja2template")

java_guard_template = env.get_template("objects/control_flow_node/java_guard.jinja2template")
java_deterministic_decision_template = env.get_template(
    "objects/control_flow_node/java_deterministic_decision.jinja2template"
)
java_non_deterministic_decision_template = env.get_template(
    "objects/control_flow_node/java_non_deterministic_decision.jinja2template"
)

java_lock_manager_template = env.get_template("locking/java_lock_manager.jinja2template")
java_lock_acquisition_block_template = env.get_template("locking/java_lock_acquisition_block.jinja2template")
java_lock_acquisition_template = env.get_template("locking/java_lock_acquisition.jinja2template")
java_lock_release_block_template = env.get_template("locking/java_lock_release_block.jinja2template")
java_lock_release_template = env.get_template("locking/java_lock_release.jinja2template")

java_object_instantiation = env.get_template("util/java_object_instantiation.jinja2template")
