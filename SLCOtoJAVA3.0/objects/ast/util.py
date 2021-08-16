from typing import Iterable

from objects.ast.models import SlcoModel, Action, Object, Initialisation, Class, StateMachine, State, Variable, Type, \
    Transition, Composite, Assignment, Expression, Primary, VariableRef, ActionRef


def __dfs__(model, self_first=False, _filter=lambda x: True) -> Iterable:
    """Iterate through all objects in the node through a DFS."""
    if self_first and _filter(model):
        yield model
    for o in model:
        if isinstance(o, Iterable):
            yield from __dfs__(o, self_first, _filter)
        elif _filter(o):
            yield o
    if not self_first and _filter(model):
        yield model


def ast_to_model(model, lookup_table: dict, parent=None):
    """Convert the given AST representations to the revised models used in the code generator."""
    type_name = type(model).__name__
    if type_name == "SLCOModel":
        result = SlcoModel(model.name)
        result.actions = [ast_to_model(a, lookup_table) for a in model.actions]
        result.channels = [c for c in model.channels]
        result.classes = [ast_to_model(c, lookup_table) for c in model.classes]
        result.objects = [ast_to_model(o, lookup_table) for o in model.objects]
        return result
    elif type_name == "Action":
        result = lookup_table[model.name] = Action(model.name)
        return result
    elif type_name == "Object":
        result = Object(model.name, lookup_table[model.type])
        result.assignments = [ast_to_model(i, lookup_table) for i in model.assignments]
        return result
    elif type_name == "Initialisation":
        result = Initialisation(lookup_table[model.left], model.right, model.rights)
        return result
    elif type_name == "Class":
        result = lookup_table[model] = Class(model.name)
        result.ports = [p for p in model.ports]
        result.variables = [ast_to_model(v, lookup_table) for v in model.variables]
        result.state_machines = [ast_to_model(sm, lookup_table, result) for sm in model.statemachines]
        return result
    elif type_name == "StateMachine":
        # noinspection PyTypeChecker
        result = StateMachine(model.name, ast_to_model(model.initialstate, lookup_table))
        result.states = [ast_to_model(s, lookup_table) for s in model.states]
        result.variables = [ast_to_model(v, lookup_table) for v in model.variables]
        for v in result.variables + parent.variables:
            lookup_table[v.name] = v
        result.transitions = [ast_to_model(t, lookup_table) for t in model.transitions]
        for v in result.variables + parent.variables:
            del lookup_table[v.name]
        return result
    elif type_name == "State":
        result = lookup_table[model] = State(model.name)
        return result
    elif type_name == "Variable":
        result = lookup_table[model] = Variable(model.name)
        result.type = ast_to_model(model.type, lookup_table)
        result.def_value = model.defvalue
        result.def_values[:] = [v for v in model.defvalues]
        return result
    elif type_name == "Type":
        result = Type(model.base, model.size)
        return result
    elif type_name == "Transition":
        result = Transition(lookup_table[model.source], lookup_table[model.target], model.priority)
        result.statements = [ast_to_model(s, lookup_table) for s in model.statements]
        return result
    elif type_name == "Composite":
        result = Composite()
        if model.guard is not None:
            result.guard = ast_to_model(model.guard, lookup_table)
        result.assignments = [ast_to_model(a, lookup_table) for a in model.assignments]
        return result
    elif type_name == "Assignment":
        result = Assignment()
        result.left = ast_to_model(model.left, lookup_table)
        result.right = ast_to_model(model.right, lookup_table)
        return result
    elif type_name in ["Expression", "ExprPrec1", "ExprPrec2", "ExprPrec3", "ExprPrec4"]:
        result = Expression(model.op)
        left = [ast_to_model(model.left, lookup_table)]
        right = [] if model.right is None else [ast_to_model(model.right, lookup_table)]
        # Note: Instead of left and right, the expression is depicted as an operation over an array of values.
        result.values = left + right
        return result
    elif type_name == "Primary":
        result = Primary(model.sign, model.value)
        if model.ref is not None:
            result.ref = ast_to_model(model.ref, lookup_table)
        if model.body is not None:
            result.body = ast_to_model(model.body, lookup_table)
        return result
    elif type_name in ["VariableRef", "ExpressionRef"]:
        result = VariableRef(lookup_table[model.var if type_name == "VariableRef" else model.ref])
        if model.index is not None:
            result.index = ast_to_model(model.index, lookup_table)
        return result
    elif type_name == "ActionRef":
        result = ActionRef(lookup_table[model.act])
        return result
    else:
        raise Exception("Received a model of unknown type %s." % type(model))


def copy_node(model, lookup_table: dict, parent=None):
    """Deep copy the given node and all of its children to new objects."""
    if isinstance(model, SlcoModel):
        target = SlcoModel(model.name)
        target.actions = [copy_node(a, lookup_table) for a in model.actions]
        target.channels = [c for c in model.channels]
        target.classes = [copy_node(c, lookup_table) for c in model.classes]
        target.objects = [copy_node(o, lookup_table) for o in model.objects]
        return target
    elif isinstance(model, Action):
        target = lookup_table[model.name] = Action(model.name)
        return target
    elif isinstance(model, Object):
        target = Object(model.name, lookup_table.get(model.type, model.type))
        target.assignments = [copy_node(i, lookup_table) for i in model.assignments]
        return target
    elif isinstance(model, Initialisation):
        result = Initialisation(lookup_table.get(model.left, model.left), model.right, model.rights)
        return result
    elif isinstance(model, Class):
        result = lookup_table[model] = Class(model.name)
        result.ports = [p for p in model.ports]
        result.variables = [copy_node(v, lookup_table) for v in model.variables]
        result.state_machines = [copy_node(sm, lookup_table, result) for sm in model.state_machines]
        return result
    elif isinstance(model, StateMachine):
        # noinspection PyTypeChecker
        result = StateMachine(model.name, copy_node(model.initial_state, lookup_table))
        result.states = [copy_node(s, lookup_table) for s in model.states]
        result.variables = [copy_node(v, lookup_table) for v in model.variables]
        for v in result.variables + parent.variables:
            lookup_table[v.name] = v
        result.transitions = [copy_node(t, lookup_table) for t in model.transitions]
        for v in result.variables + parent.variables:
            del lookup_table[v.name]
        return result
    elif isinstance(model, State):
        result = lookup_table[model] = State(model.name)
        return result
    elif isinstance(model, Variable):
        result = lookup_table[model] = Variable(model.name)
        result.lock_id = model.lock_id
        result.type = copy_node(model.type, lookup_table)
        result.def_value = model.def_value
        result.def_values = [v for v in model.def_values]
        return result
    elif isinstance(model, Type):
        result = Type(model.base, model.size)
        return result
    elif isinstance(model, Transition):
        result = Transition(lookup_table[model.source], lookup_table[model.target], model.priority)
        result.statements = [copy_node(s, lookup_table) for s in model.statements]
        return result
    elif isinstance(model, Composite):
        result = Composite()
        result.exclude_statement = model.exclude_statement
        result.produced_statement = model.produced_statement
        result.original_statement = copy_node(model.original_statement, lookup_table)
        if model.guard is not None:
            result.guard = copy_node(model.guard, lookup_table)
        result.assignments = [copy_node(a, lookup_table) for a in model.assignments]
        return result
    elif isinstance(model, Assignment):
        result = Assignment()
        result.exclude_statement = model.exclude_statement
        result.produced_statement = model.produced_statement
        result.original_statement = copy_node(model.original_statement, lookup_table)
        result.left = copy_node(model.left, lookup_table)
        result.right = copy_node(model.right, lookup_table)
        return result
    elif isinstance(model, Expression):
        result = Expression(model.op)
        result.exclude_statement = model.exclude_statement
        result.produced_statement = model.produced_statement
        result.original_statement = copy_node(model.original_statement, lookup_table)
        result.values = [copy_node(v, lookup_table) for v in model.values]
        return result
    elif isinstance(model, Primary):
        result = Primary(model.sign, model.value)
        result.exclude_statement = model.exclude_statement
        result.produced_statement = model.produced_statement
        result.original_statement = copy_node(model.original_statement, lookup_table)
        if model.ref is not None:
            result.ref = copy_node(model.ref, lookup_table)
        if model.body is not None:
            result.body = copy_node(model.body, lookup_table)
        return result
    elif isinstance(model, VariableRef):
        result = VariableRef(lookup_table.get(model.var, model.var))
        if model.index is not None:
            result.index = copy_node(model.index, lookup_table)
        return result
    elif isinstance(model, ActionRef):
        result = ActionRef(lookup_table.get(model.act, model.act))
        result.exclude_statement = model.exclude_statement
        result.produced_statement = model.produced_statement
        result.original_statement = copy_node(model.original_statement, lookup_table)
        return result
    else:
        if model is not None:
            print("Object", model, "could not be copied.")
        return None
