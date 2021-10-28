# Initialize the template engine.
import jinja2 as jinja2


# UTIL FUNCTIONS
from objects.ast.models import SlcoModel, Object, Class


def comma_separated_list(model):
    """Construct a comma separated list of the given iterable"""
    return ", ".join(model)


# MODEL RENDERING FUNCTIONS
def render_model(model: SlcoModel):
    """Render the SLCO model as Java code"""
    return java_model_template.render(model=model)


def render_lock_manager():
    """Render the lock manager of the model."""
    return java_lock_manager.render()


def render_object_instantiation(model: Object):
    """Render the instantiation of the given object."""
    arguments = []
    for i, a in enumerate(model.initial_values):
        v = model.type.variables[i]
        if v.is_array:
            arguments.append("new %s[]{%s}" % ("boolean" if v.is_boolean else "int", ", ".join(a)))
        else:
            arguments.append(a)

    return java_object_instantiation.render(
        name=model.name,
        arguments=arguments
    )


def render_class(model: Class):
    """Render the SLCO class as Java code"""
    return java_class_template.render(model=model)


def render_state_machine(model):
    """Render the SLCO state machine as Java code"""
    return java_state_machine_template.render(model=model)


env = jinja2.Environment(
    loader=jinja2.FileSystemLoader('jinja2_templates'),
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=['jinja2.ext.loopcontrols', 'jinja2.ext.do']
)

# Register the rendering filters
env.filters['render_class'] = render_class
env.filters['render_lock_manager'] = render_lock_manager
env.filters['render_state_machine'] = render_state_machine

# Register the utility filters
env.filters['comma_separated_list'] = comma_separated_list
# env.filters['get_instruction'] = get_instruction
# env.filters['get_guard_statement'] = get_guard_statement
# env.filters['get_decision_structure'] = get_decision_structure
# env.filters['get_lock_id_list'] = get_lock_id_list

# Load the Java templates.
java_model_template = env.get_template('objects/java_model.jinja2template')
java_class_template = env.get_template('objects/java_class.jinja2template')
java_state_machine_template = env.get_template('objects/java_state_machine.jinja2template')
java_transition_template = env.get_template('objects/java_transition.jinja2template')

java_assignment_template = env.get_template('objects/java_assignment.jinja2template')
java_expression_template = env.get_template('objects/java_expression.jinja2template')
java_composite_template = env.get_template('objects/java_composite.jinja2template')

java_lock_manager = env.get_template('locking/java_lock_manager.jinja2template')
java_object_instantiation = env.get_template('util/java_object_instantiation.jinja2template')
