from __future__ import annotations

from collections import defaultdict
from typing import Union, List, Optional, Iterator, Dict, Set
from networkx import DiGraph
from objects.ast.interfaces import Preprocessable, DFS, Evaluable, Lockable, Copyable
from preprocessing.statement_beautification import beautify
from preprocessing.statement_preprocessing import preprocess
from preprocessing.statement_simplification import simplify
from util.smt import operator_mapping

import libraries.slcolib as slco2
import networkx as nx
import operator


# TODO: redo the entire pre-processing procedure.


# SLCO TYPES
class SLCOModel(Preprocessable, DFS):
    """
    An object representing the encompassing model in the SLCO framework.
    """
    def __init__(self, name: str) -> None:
        self._actions: List[Action] = []
        self.channels: List[str] = []
        self._classes: List[Class] = []
        self.name = name
        self._objects: List[Object] = []

    # noinspection DuplicatedCode
    @classmethod
    def from_model(cls, model) -> SLCOModel:
        """Convert the given model instance to the revised instance."""
        lookup_table = dict()

        result = cls(model.name)
        result.actions = [Action.from_model(a, lookup_table) for a in model.actions]
        result.channels = [c for c in model.channels]
        result.classes = [Class.from_model(c, lookup_table) for c in model.classes]
        result.objects = [Object.from_model(o, lookup_table) for o in model.objects]
        result.preprocess()

        return result

    def __repr__(self) -> str:
        return "SLCOModel:%s" % self.name

    def __iter__(self) -> Iterator[Class]:
        """Iterate through all objects part of the AST structure."""
        for c in self.classes:
            yield c

    @property
    def actions(self) -> List[Action]:
        return self._actions

    @actions.setter
    def actions(self, val) -> None:
        self._actions[:] = val
        for v in self._actions:
            v.parent = self

    @property
    def classes(self) -> List[Class]:
        return self._classes

    @classes.setter
    def classes(self, val) -> None:
        self._classes[:] = val
        for v in self._classes:
            v.parent = self

    @property
    def objects(self) -> List[Object]:
        return self._objects

    @objects.setter
    def objects(self, val) -> None:
        self._objects[:] = val
        for v in self._objects:
            v.parent = self


class Action:
    """
    An object representing actions in the SLCO framework.
    """
    def __init__(self, name: str) -> None:
        self.parent = None
        self.name = name

    @classmethod
    def from_model(cls, model, lookup_table: dict) -> Action:
        result = lookup_table[model.name] = cls(model.name)
        return result

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Action):
            return self.name == o.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)


class Object:
    """
    An object representing class instantiations in the SLCO framework.
    """
    type: Class

    def __init__(self, name: str) -> None:
        self.parent = None
        self._assignments: List[Initialisation] = []
        self.name = name

    @classmethod
    def from_model(cls, model, lookup_table: dict) -> Object:
        """Convert the given model instance to the revised instance."""
        result = cls(model.name)
        result.assignments = [Initialisation.from_model(i, lookup_table) for i in model.assignments]
        result.type = lookup_table[model.type]
        return result

    def __repr__(self) -> str:
        return "Object:%s" % self.name

    @property
    def assignments(self) -> List[Initialisation]:
        return self._assignments

    @assignments.setter
    def assignments(self, val) -> None:
        self._assignments[:] = val
        for v in self._assignments:
            v.parent = self


class Initialisation:
    """
    An object representing variable initialisations in the SLCO framework.
    """
    left: Variable
    right: Union[int, bool]
    rights: Union[List[int], List[bool]]

    def __init__(self) -> None:
        self.parent = None

    @classmethod
    def from_model(cls, model, lookup_table: dict) -> Initialisation:
        """Convert the given model instance to the revised instance."""
        result = cls()
        result.left = lookup_table[model.left]
        result.right = model.right
        result.rights = model.rights
        return result

    def __repr__(self) -> str:
        return "%s := %s" % (self.left, self.rights if self.right is None else self.right)


class Class(Preprocessable, DFS):
    """
    An object representing classes in the SLCO framework.
    """
    def __init__(self, name: str) -> None:
        self.parent = None
        self.name = name
        self.ports: List[str] = []
        self._state_machines: List[StateMachine] = []
        self._variables: List[Variable] = []

    # noinspection DuplicatedCode
    @classmethod
    def from_model(cls, model, lookup_table: dict) -> Class:
        """Convert the given model instance to the revised instance."""
        result = lookup_table[model] = cls(model.name)
        result.ports = [p for p in model.ports]
        result.variables = [Variable.from_model(v, lookup_table) for v in model.variables]
        result.state_machines = [StateMachine.from_model(sm, result, lookup_table) for sm in model.statemachines]
        return result

    def get_weighted_class_variable_dependency_graph(self) -> DiGraph:
        """Get a weighted (nodes + edges) class variable dependency graph for all statements within the class."""
        graph = nx.DiGraph()
        for sm in self.state_machines:
            for g in sm.state_to_transitions.values():
                for t in g:
                    for s in t.statements:
                        # The weight increase should only be applied once per object, since locking is only done once.
                        references = set(s.get_class_variable_references())
                        used_variables = set(r.var for r in references)
                        for v in used_variables:
                            if graph.has_node(v):
                                graph.nodes[v]["weight"] += 1
                            else:
                                graph.add_node(v, weight=1)

                        inserted_variable_pairs = set()
                        while len(references) > 0:
                            source = references.pop()
                            if source.index:
                                target_references = source.index.get_class_variable_references()
                                for target in target_references:
                                    if (source.var, target.var) not in inserted_variable_pairs:
                                        inserted_variable_pairs.add((source.var, target.var))
                                        if graph.has_edge(source.var, target.var):
                                            graph[source.var][target.var]["weight"] += 1
                                        else:
                                            graph.add_edge(source.var, target.var, weight=1)
                                        references.update(target_references)
        return graph

    def __repr__(self) -> str:
        return "Class:%s" % self.name

    def __iter__(self) -> Iterator[StateMachine]:
        """Iterate through all objects part of the AST structure."""
        for sm in self.state_machines:
            yield sm

    @property
    def state_machines(self) -> List[StateMachine]:
        return self._state_machines

    @state_machines.setter
    def state_machines(self, val) -> None:
        self._state_machines[:] = val
        for v in self._state_machines:
            v.parent = self

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    @variables.setter
    def variables(self, val) -> None:
        self._variables[:] = val
        for v in self._variables:
            v.parent = self


class StateMachine(Preprocessable, DFS):
    """
    An object representing state machines in the SLCO framework.
    """
    _initial_state: State

    def __init__(self, name: str) -> None:
        self.parent = None
        self.name = name
        self._states: List[State] = []
        self._variables: List[Variable] = []
        self._transitions: List[Transition] = []
        self.state_to_transitions: Dict[State, List[Transition]] = defaultdict(list)

    @classmethod
    def from_model(cls, model, parent, lookup_table: dict) -> StateMachine:
        """Convert the given model instance to the revised instance."""
        result = cls(model.name)
        result._initial_state = State.from_model(model.initialstate, lookup_table)
        result.states = [State.from_model(s, lookup_table) for s in model.states]
        result.variables = [Variable.from_model(v, lookup_table) for v in model.variables]

        for v in result.variables + parent.variables:
            lookup_table[v.name] = v
        result.transitions = [Transition.from_model(t, lookup_table) for t in model.transitions]
        for v in result.variables + parent.variables:
            del lookup_table[v.name]

        return result

    def __repr__(self) -> str:
        return "StateMachine:%s" % self.name

    def __iter__(self) -> Iterator[Transition]:
        """Iterate through all objects part of the AST structure."""
        for t in self.transitions:
            yield t

    def preprocess(self) -> None:
        """Preprocess the structure of the AST to simplify and generalize."""
        super(StateMachine, self).preprocess()
        self.transitions = [preprocess(s) for s in self.transitions]
        self.transitions = [simplify(s) for s in self.transitions]
        self.transitions = [beautify(s) for s in self.transitions]

        for t in self.transitions:
            self.state_to_transitions[t.source].append(t)

    @property
    def initial_state(self) -> State:
        return self._initial_state

    @property
    def states(self) -> List[State]:
        """Get the list of states excluding the initial state."""
        return self._states

    @property
    def i_states(self) -> List[State]:
        """Get the list of states including the initial state."""
        return [self._initial_state] + self._states

    @states.setter
    def states(self, val) -> None:
        self._states[:] = val
        for v in self._states:
            v.parent = self

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    @variables.setter
    def variables(self, val) -> None:
        self._variables[:] = val
        for v in self._variables:
            v.parent = self

    @property
    def transitions(self) -> List[Transition]:
        return self._transitions

    @transitions.setter
    def transitions(self, val) -> None:
        self._transitions[:] = val
        for v in self._transitions:
            v.parent = self


class State:
    """
    An object representing states in the SLCO framework.
    """
    def __init__(self, name: str) -> None:
        self.parent = None
        self.name = name

    @classmethod
    def from_model(cls, model, lookup_table: dict) -> State:
        """Convert the given model instance to the revised instance."""
        result = lookup_table[model] = cls(model.name)
        return result

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, o: object) -> bool:
        if isinstance(o, State):
            return self.name == o.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)


class Variable(Evaluable):
    """
    An object representing variables in the SLCO framework.
    """
    type: Type
    lock_id: int

    def __init__(self, name: str, _type: Optional[Type] = None) -> None:
        self.parent = None
        self.name = name
        self.lock_id = -1
        self.def_value: Optional[Union[int, bool]] = None
        self.def_values: Union[List[int], List[bool]] = []
        if _type is not None:
            self.type = _type

    @classmethod
    def from_model(cls, model: slco2.Variable, lookup_table) -> Variable:
        """Convert the given model instance to the revised instance."""
        result = lookup_table[model] = cls(model.name)
        result.def_value = model.defvalue
        result.def_values[:] = [v for v in model.defvalues]
        result.type = Type.from_model(model.type)
        return result

    def __repr__(self) -> str:
        if self.is_class_variable:
            return "%s': %s" % (self.name, self.type)
        else:
            return "%s: %s" % (self.name, self.type)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Variable):
            return self.name == o.name and self.type == o.type
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.type))

    @property
    def is_array(self) -> bool:
        return self.type.is_array

    @property
    def is_boolean(self) -> bool:
        return self.type.is_boolean

    @property
    def is_byte(self) -> bool:
        return self.type.is_byte

    @property
    def is_class_variable(self) -> bool:
        """Returns ``True`` when the parent object is a class, ``False`` otherwise."""
        return isinstance(self.parent, Class)


class Type:
    """
    An object representing variable types in the SLCO framework.
    """
    def __init__(self, base: str, size: int) -> None:
        self.parent = None
        self.base = base
        self.size = size

    @classmethod
    def from_model(cls, model: slco2.Type) -> Type:
        """Convert the given model instance to the revised instance."""
        return cls(model.base, model.size)

    def __repr__(self) -> str:
        base_abbreviation = "bool" if self.is_boolean else "byte" if self.is_byte else "int"
        if self.is_array:
            return "%s[%s]" % (base_abbreviation, self.size)
        else:
            return base_abbreviation

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Type):
            return self.base == o.base and self.size == o.size
        return False

    def __hash__(self) -> int:
        return hash((self.base, self.size))

    @property
    def is_array(self) -> bool:
        """Returns ``True`` when the type is an array, ``False`` otherwise."""
        return self.size > 0

    @property
    def is_boolean(self) -> bool:
        """Returns ``True`` when the type is a Boolean, ``False`` otherwise."""
        return self.base == "Boolean"

    @property
    def is_byte(self) -> bool:
        """Returns ``True`` when the type is a Byte, ``False`` otherwise."""
        return self.base == "Byte"


class Transition(DFS):
    """
    An object representing a guarded transition in the SLCO framework.

    Notes for after simplification:
        - Transitions always start with an expression or composite. If missing, a true expression will be prepended.
        - If the first statement is a composite with no guard, a true expression is prepended to the statement list.
        - All statements are of the types Composite, Assignment, Expression, ActionRef or Primary.
        - Superfluous Expression and Primary statements are removed.
        - Composites that only contain a guard are automatically converted to an Expression instead.
    """
    source: State
    target: State

    def __init__(self, priority: int = 0) -> None:
        self.parent = None
        self.priority = priority
        self._statements = []

    # noinspection PyArgumentList
    @classmethod
    def from_model(cls, model, lookup_table: dict) -> Transition:
        """Convert the given model instance to the revised instance."""
        result = cls(model.priority)
        result.source = lookup_table[model.source]
        result.target = lookup_table[model.target]
        result.statements = [_class_conversion_table[type(s)](s, lookup_table) for s in model.statements]
        return result

    def __repr__(self) -> str:
        transition_repr = "%s: %s -> %s {" % (self.priority, self.source, self.target)
        for s in self.statements:
            transition_repr += "\n\t%s;" % s
        transition_repr += "\n}"
        return transition_repr

    def __iter__(self) -> Iterator[Union[Expression, Composite, Assignment, Primary]]:
        """Iterate through all objects part of the AST structure."""
        for s in self.statements:
            yield s

    @property
    def guard(self) -> Union[Expression, Primary, Composite]:
        """Get the expression that acts as the guard expression of the transition."""
        return self.statements[0]

    @property
    def statements(self):
        return self._statements

    @statements.setter
    def statements(self, val) -> None:
        self._statements[:] = val
        for v in self._statements:
            v.parent = self


class Composite(Evaluable, Lockable):
    """
    An object representing composite statements in the SLCO framework.

    Notes for after simplification:
        - Composites always have a guard. If none is present previously, a true expression is set as the guard instead.
        - Superfluous guard statements are simplified to a true expression when appropriate.
    """
    def __init__(self, guard=None, assignments=None) -> None:
        self.parent = None
        self._guard = None
        self._assignments: List[Assignment] = []

        # A composite will always have a guard. If none is given, set it to a true primary.
        if guard is not None:
            self.guard = guard
        else:
            self.guard = Primary(target=True)

        if assignments is not None:
            self.assignments = assignments

    @classmethod
    def from_model(cls, model: slco2.Composite, lookup_table: dict) -> Composite:
        """Convert the given model instance to the revised instance."""
        result = cls()
        if model.guard is not None:
            result.guard = Expression.from_model(model.guard, lookup_table)
        result.assignments = [Assignment.from_model(a, lookup_table) for a in model.assignments]
        return result

    def get_variable_references(self) -> Set[VariableRef]:
        """Get a list of all the variables that have been referenced to by the statement."""
        variable_references = self.guard.get_variable_references()

        # The assignment statements may alter the values of the variables, and hence, a rewrite table is needed.
        rewrite_rules = dict()
        for a in self.assignments:
            # Start by rewriting the assignment.
            rewritten_assignment = a.create_copy(rewrite_rules)

            # Get the variable references for the statement and apply the rewrite rules.
            variable_references.update(rewritten_assignment.get_variable_references())

            # Add a rewrite rule for the current assignment.
            rewrite_rules[rewritten_assignment.left] = rewritten_assignment.right

        return variable_references

    def __repr__(self) -> str:
        statements = [self.guard] if self.guard is not None else []
        statements += [s for s in self.assignments]
        return "[%s]" % "; ".join(str(s) for s in statements)

    def __iter__(self) -> Iterator[Union[Expression, Primary, Assignment]]:
        """Iterate through all objects part of the AST structure."""
        if self.guard is not None:
            yield self.guard
        for s in self.assignments:
            yield s

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Composite):
            if len(self.assignments) != len(o.assignments):
                return False
            for i in range(0, len(self.assignments)):
                if self.assignments[i] != o.assignments[i]:
                    return False
            return self.guard.is_equivalent(o.guard)
        return False

    @property
    def guard(self):
        return self._guard

    @guard.setter
    def guard(self, val):
        self._guard = val
        if val:
            val.parent = self

    @property
    def assignments(self) -> List[Assignment]:
        return self._assignments

    @assignments.setter
    def assignments(self, val) -> None:
        self._assignments[:] = val
        for v in self._assignments:
            v.parent = self


class Assignment(Lockable, Copyable):
    """
    An object representing assignment statements in the SLCO framework.
    """
    def __init__(self, left=None, right=None) -> None:
        self.parent = None
        self._right = None
        self._left = None

        if left is not None:
            self.left = left
        if right is not None:
            self.right = right

    # noinspection PyArgumentList
    @classmethod
    def from_model(cls, model: slco2.Assignment, lookup_table: dict) -> Assignment:
        """Convert the given model instance to the revised instance."""
        result = cls()
        result.left = _class_conversion_table[type(model.left)](model.left, lookup_table)
        result.right = _class_conversion_table[type(model.right)](model.right, lookup_table)
        return result

    def create_copy(self, rewrite_rules: dict, is_first: bool = True) -> Assignment:
        """Create a copy of the expression with the given rewrite rules applied."""
        result = Assignment()
        result.left = self.left.create_copy(rewrite_rules, False)
        result.right = self.right.create_copy(rewrite_rules, False)
        if is_first:
            result = simplify(result)
            result = beautify(result)
        return result

    def __repr__(self) -> str:
        return "%s := %s" % (self.left, self.right)

    def __iter__(self) -> Iterator[Union[VariableRef, Expression, Primary]]:
        """Iterate through all objects part of the AST structure."""
        yield self.left
        yield self.right

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Assignment):
            return self.left.is_equivalent(o.left) and self.right.is_equivalent(o.right)
        return False

    @property
    def left(self) -> VariableRef:
        return self._left

    @left.setter
    def left(self, val) -> None:
        self._left = val
        if val:
            val.parent = self

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, val) -> None:
        self._right = val
        if val:
            val.parent = self


# Several operators differ when using smt. Overwrite those.
resolution_operator_mapping = operator_mapping.copy()
resolution_operator_mapping["xor"] = operator.xor
resolution_operator_mapping["or"] = operator.or_
resolution_operator_mapping["and"] = operator.and_


# Several operators have different presentations. Normalize them.
operator_normalizations = {
    "&&": "and",
    "||": "or",
    "<>": "!=",
}


class Expression(Evaluable, Lockable, Copyable):
    """
    An object representing expression statements in the SLCO framework.

    Notes:
        - The ordering of variables in the expressions are not changed.

    TODO:
        - Expressions may contain bounds on variables used in array accesses. Map these in a graph.
    """
    def __init__(self, op: str, values=None) -> None:
        self.parent = None
        self.op = operator_normalizations.get(op, op)

        # Note: Instead of left and right, the expression is depicted as an operation over an array of values.
        self._values = []

        if values is not None:
            self.values = values

    # noinspection PyArgumentList
    @classmethod
    def from_model(
            cls,
            model: Union[slco2.Expression, slco2.ExprPrec1, slco2.ExprPrec2, slco2.ExprPrec3, slco2.ExprPrec4],
            lookup_table: dict
    ) -> Expression:
        """Convert the given model instance to the revised instance."""
        result = cls(model.op)
        left = [_class_conversion_table[type(model.left)](model.left, lookup_table)]
        right = [] if model.right is None else [_class_conversion_table[type(model.right)](model.right, lookup_table)]
        result.values = left + right
        return result

    def create_copy(self, rewrite_rules: dict, is_first: bool = True) -> Expression:
        """Create a copy of the expression with the given rewrite rules applied."""
        result = Expression(self.op)
        result.values = [v.create_copy(rewrite_rules, False) for v in self.values]
        if is_first:
            result = simplify(result)
            result = beautify(result)
        return result

    def __repr__(self) -> str:
        return (" %s " % self.op).join(str(v) for v in self.values)

    def __iter__(self):
        """Iterate through all objects part of the AST structure."""
        for v in self.values:
            yield v

    def __eq__(self, o: object) -> bool:
        if isinstance(o, (Primary, Expression)):
            return self.is_equivalent(o)
        return False

    def __hash__(self) -> int:
        # return super(Expression, self).__hash__()
        return 0

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, val) -> None:
        self._values[:] = val
        for v in self._values:
            v.parent = self


class Primary(Evaluable, Lockable, Copyable):
    """
    An object representing primary values in the SLCO framework.
    """
    def __init__(
            self,
            sign: str = "",
            value: Optional[Union[int, bool]] = None,
            target=None,
    ) -> None:
        self.parent = None
        self._body: Optional[Expression, Primary] = None
        self._ref: Optional[VariableRef] = None
        self.sign = sign
        self.value = value

        if target is not None:
            if isinstance(target, Primary):
                # Copy over all values.
                self.body = target._body
                self.ref = target._ref
                self.value = target.value
                if sign == target.sign:
                    self.sign = ""
                else:
                    self.sign = target.sign or sign
            elif isinstance(target, Expression):
                self.body = target
            elif isinstance(target, VariableRef):
                self.ref = target
            elif isinstance(target, bool):
                self.value = target
            elif target < 0:
                self.value = -target
                self.sign = "-"
            else:
                self.value = target

    @classmethod
    def from_model(cls, model: slco2.Primary, lookup_table: dict) -> Primary:
        """Convert the given model instance to the revised instance."""
        result = cls(model.sign, model.value)
        if model.ref is not None:
            result.ref = VariableRef.from_model(model.ref, lookup_table)
        if model.body is not None:
            result.body = Expression.from_model(model.body, lookup_table)
        return result

    def create_copy(self, rewrite_rules: dict, is_first: bool = True) -> Primary:
        """Create a copy of the expression with the given rewrite rules applied."""
        result = Primary(sign=self.sign, value=self.value)
        if self.body:
            result.body = self.body.create_copy(rewrite_rules, False)
        if self.ref:
            if self.ref in rewrite_rules:
                result.body = rewrite_rules[self.ref].create_copy(result, dict(), False)
            else:
                result.ref = self.ref.create_copy(rewrite_rules, False)
        if is_first:
            result = simplify(result)
            result = beautify(result)
        return result

    def __repr__(self) -> str:
        if self.value is not None:
            exp_str = str(self.value).lower()
        elif self.ref is not None:
            exp_str = "%s" % self.ref
        else:
            exp_str = "(%s)" % self.body
        return ("!%s" if self.sign == "not" else self.sign + "%s") % exp_str

    def __iter__(self):
        """Iterate through all objects part of the AST structure."""
        if self.ref is not None:
            yield self.ref
        if self.body is not None:
            yield self.body

    def __eq__(self, o: object) -> bool:
        if isinstance(o, (Primary, Expression)):
            return self.is_equivalent(o)
        return False

    def __hash__(self) -> int:
        # return super(Primary, self).__hash__()
        return 0

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, val):
        self._body = val
        if val is not None:
            val.parent = self

    @property
    def ref(self) -> Optional[VariableRef]:
        return self._ref

    @ref.setter
    def ref(self, val):
        self._ref = val
        if val is not None:
            val.parent = self

    @property
    def signed_value(self) -> Optional[Union[int, bool]]:
        """Get the value of the primary with the given sign applied."""
        if isinstance(self.value, int):
            if self.sign == "-":
                return -self.value
            else:
                return self.value
        elif isinstance(self.value, bool):
            if self.sign == "not":
                return not self.value
            else:
                return self.value
        return self.value


class VariableRef(Evaluable, DFS, Copyable):
    """
    An object representing references to variables in the SLCO framework.

    Notes:
        - VariableRef objects do not necessarily need a parent. Hence, the parent can be ``None``.
    """
    def __init__(self, var=None, index=None) -> None:
        self.parent = None
        self.var = var
        self._index = None
        if index is not None:
            self.index = index

    @classmethod
    def from_model(
            cls, model: Union[slco2.VariableRef, slco2.ExpressionRef], lookup_table: dict
    ) -> VariableRef:
        """Convert the given model instance to the revised instance."""
        result = cls()
        if model.index is not None:
            result.index = Expression.from_model(model.index, lookup_table)
        result.var = lookup_table[model.var if isinstance(model, slco2.VariableRef) else model.ref]
        return result

    def create_copy(self, rewrite_rules: dict, is_first: bool = True) -> VariableRef:
        """Create a copy of the expression with the given rewrite rules applied."""
        result = VariableRef()
        result.var = self.var
        if self.index:
            result.index = self.index.create_copy(result, rewrite_rules, False)
        if is_first:
            result = simplify(result)
            result = beautify(result)
        return result

    def __repr__(self) -> str:
        var_str = self.var.name
        if self.var.is_class_variable:
            var_str += "'"
        if self.index is not None:
            var_str += "[%s]" % self.index
        return var_str

    def __iter__(self):
        """Iterate through all objects part of the AST structure."""
        if self.index is not None:
            yield self.index

    def __eq__(self, o: object) -> bool:
        if isinstance(o, VariableRef):
            return self.var == o.var and self.is_equivalent(o)
        return False

    def __hash__(self) -> int:
        # A weak hash only containing the target variable is required, since indices are too complex to hash reliably.
        return hash(self.var)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, val):
        self._index = val
        if val:
            val.parent = self


class ActionRef:
    """
    An object representing references to model actions in the SLCO framework.
    """
    act: Action

    def __init__(self) -> None:
        self.parent = None

    @classmethod
    def from_model(cls, model: slco2.ActionRef, lookup_table: dict) -> ActionRef:
        """Convert the given model instance to the revised instance."""
        result = cls()
        result.act = lookup_table[model.act]
        return result

    def __repr__(self) -> str:
        return "%s" % self.act

    def __eq__(self, o: object) -> bool:
        if isinstance(o, ActionRef):
            return self.act == o.act
        return False

    def __hash__(self) -> int:
        return hash(self.act)


# A lookup dictionary that allows for the dynamic conversion of slcolib types to extended types.
_class_conversion_table = {
    slco2.Composite: Composite.from_model,
    slco2.Assignment: Assignment.from_model,
    slco2.Expression: Expression.from_model,
    slco2.ExprPrec1: Expression.from_model,
    slco2.ExprPrec2: Expression.from_model,
    slco2.ExprPrec3: Expression.from_model,
    slco2.ExprPrec4: Expression.from_model,
    slco2.Primary: Primary.from_model,
    slco2.VariableRef: VariableRef.from_model,
    slco2.ExpressionRef: VariableRef.from_model,
    slco2.ActionRef: ActionRef.from_model,
}
