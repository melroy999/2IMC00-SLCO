from __future__ import annotations

from collections import defaultdict
from typing import Union, List, Optional, Iterator, Dict, Set, TYPE_CHECKING

import networkx as nx

import settings
from objects.ast.interfaces import SlcoNode, SlcoStructuralNode, SlcoEvaluableNode, SlcoStatementNode, SlcoLockableNode

if TYPE_CHECKING:
    from objects.locking.models import Lock, AtomicNode


# SLCO TYPES
class SlcoModel(SlcoStructuralNode):
    """
    An object representing the encompassing model in the SLCO framework.
    """
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self._actions: List[Action] = []
        self.channels: List[str] = []
        self._classes: List[Class] = []
        self._objects: List[Object] = []

    def __repr__(self) -> str:
        return f"SLCOModel:{self.name}"

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


class Action(SlcoNode):
    """
    An object representing actions in the SLCO framework.
    """
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"Action:{self.name}"

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Action):
            return self.name == o.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)


class Object(SlcoNode):
    """
    An object representing class instantiations in the SLCO framework.
    """
    def __init__(self, name: str, _type: Class) -> None:
        super().__init__()
        self.name = name
        self.type = _type
        self._assignments: List[Initialisation] = []
        self.initial_values = []

    def __repr__(self) -> str:
        return f"Object:{self.name}"

    @property
    def assignments(self) -> List[Initialisation]:
        return self._assignments

    @assignments.setter
    def assignments(self, val) -> None:
        self._assignments[:] = val
        for v in self._assignments:
            v.parent = self


class Initialisation(SlcoNode):
    """
    An object representing variable initialisations in the SLCO framework.
    """
    def __init__(self, left: Variable, right: Union[int, bool], rights: Union[List[int], List[bool]]) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.rights = rights

    def __repr__(self) -> str:
        return f"{self.left} := {self.rights if self.right is None else self.right}"


class Class(SlcoStructuralNode):
    """
    An object representing classes in the SLCO framework.
    """
    weighted_variable_dependency_graph: nx.DiGraph = None
    weighted_class_variable_dependency_graph: nx.DiGraph = None

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.ports: List[str] = []
        self._state_machines: List[StateMachine] = []
        self._variables: List[Variable] = []

    def __repr__(self) -> str:
        return f"Class:{self.name}"

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

    @property
    def max_lock_id(self) -> int:
        return max((v.lock_id + v.type.size for v in self.variables), default=0)


class StateMachine(SlcoStructuralNode):
    """
    An object representing state machines in the SLCO framework.
    """

    def __init__(self, name: str, initial_state: State) -> None:
        super().__init__()
        self.name = name
        self._initial_state = initial_state
        self._states: List[State] = []
        self._variables: List[Variable] = []
        self._transitions: List[Transition] = []
        self.state_to_transitions: Dict[State, List[Transition]] = defaultdict(list)
        self.state_to_decision_node: Dict[State, DecisionNode] = dict()

        # Locking information.
        self.lock_ids_list_size: int = 0
        self.target_locks_list_size: int = 0

    def __repr__(self) -> str:
        return f"StateMachine:{self.name}"

    def __iter__(self) -> Iterator[Transition]:
        """Iterate through all objects part of the AST structure."""
        for t in self.transitions:
            yield t

    @property
    def initial_state(self) -> State:
        return self._initial_state

    @property
    def states(self) -> List[State]:
        """Get the list of states, including the initial state as the first element."""
        return self._states

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


class State(SlcoNode):
    """
    An object representing states in the SLCO framework.
    """
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, o: object) -> bool:
        if isinstance(o, State):
            return self.name == o.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)


class Variable(SlcoEvaluableNode):
    """
    An object representing variables in the SLCO framework.
    """
    def __init__(self, name: str, _type: Optional[Type] = None) -> None:
        super().__init__()
        self.name = name
        self.lock_id = -1
        self.type = _type
        self.def_value: Optional[Union[int, bool]] = None
        self.def_values: Union[List[int], List[bool]] = []

    def __repr__(self) -> str:
        return f"{self.name}: {self.type}"

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


class Type(SlcoNode):
    """
    An object representing variable types in the SLCO framework.
    """
    def __init__(self, base: str, size: int) -> None:
        super().__init__()
        self.base = base
        self.size = size
        self.size_n = 0

    def __repr__(self) -> str:
        base_abbreviation = "bool" if self.is_boolean else "byte" if self.is_byte else "int"
        if self.is_array:
            return f"{base_abbreviation}[{self.size}]"
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


class Transition(SlcoStructuralNode):
    """
    An object representing a guarded transition in the SLCO framework.

    Notes for after simplification:
        - Transitions always start with an expression or composite. If missing, a true expression will be prepended.
        - If the first statement is a composite with no guard, a true expression is prepended to the statement list.
        - All statements are of the types Composite, Assignment, Expression, ActionRef or Primary.
        - Superfluous Expression and Primary statements are marked as obsolete.
        - Composites that only contain a guard are automatically converted to an Expression instead.
    """
    def __init__(self, source: State, target: State, priority: int = 0) -> None:
        super().__init__()
        self.source = source
        self.target = target
        self.priority = priority
        self._statements: List[SlcoStatementNode] = []

        # Decision structure info.
        self.id: int = -1
        self.is_excluded = False

    def __repr__(self) -> str:
        return f"{'(Excluded) ' if self.is_excluded else ''}(p:{self.priority}, id:{self.id}) " \
               f"| {self.source} -> {self.target} | {' | '.join(str(s) for s in self.statements)}"

    def __iter__(self) -> Iterator[Union[Expression, Composite, Assignment, Primary]]:
        """Iterate through all objects part of the AST structure."""
        for s in self.statements:
            yield s

    @property
    def guard(self) -> Union[Expression, Primary, Composite]:
        """Get the expression that acts as the guard expression of the transition."""
        # noinspection PyTypeChecker
        return self.statements[0]

    @property
    def statements(self):
        return self._statements

    @statements.setter
    def statements(self, val) -> None:
        self._statements[:] = val
        for v in self._statements:
            v.parent = self

    @property
    def locking_atomic_node(self) -> AtomicNode:
        """
        Get the atomic node of the transition's guard statement.
        """
        return self.guard.locking_atomic_node

    @property
    def get_isolated_atomic_nodes(self) -> List[AtomicNode]:
        """
        Get the atomic node of the transition's secondary statements.
        """
        return [s.locking_atomic_node for s in self.statements[1:]]

    @property
    def used_variables(self) -> Set[Variable]:
        """
        Get the variables used within the transition's guard statement.
        """
        return self.locking_atomic_node.used_variables

    @property
    def location_sensitive_locks(self) -> List[Lock]:
        """
        Get the location sensitive locks used within the transition's guard statement.
        """
        return self.locking_atomic_node.location_sensitive_locks


class Composite(SlcoStatementNode, SlcoEvaluableNode):
    """
    An object representing composite statements in the SLCO framework.

    Notes for after simplification:
        - Composites always have a guard. If none is present previously, a true expression is set as the guard instead.
        - Superfluous guard statements are simplified to a true expression when appropriate.
    """
    def __init__(self, guard=None, assignments=None) -> None:
        super().__init__()
        self._guard = None
        self._assignments: List[Assignment] = []

        if guard is not None:
            self.guard = guard
        if assignments is not None:
            self.assignments = assignments

    def __repr__(self) -> str:
        statements = [self.guard] if self.guard is not None else []
        statements += [s for s in self.assignments]
        return f"[{'; '.join(str(s) for s in statements)}]"

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


class Assignment(SlcoStatementNode):
    """
    An object representing assignment statements in the SLCO framework.
    """
    def __init__(self) -> None:
        super().__init__()
        self._left = None
        self._right = None

    def __repr__(self) -> str:
        return f"{self.left} := {self.right}"

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


# Several operators have different presentations. Normalize them.
operator_normalizations = {
    "&&": "and",
    "||": "or",
    "<>": "!=",
}


class Expression(SlcoStatementNode, SlcoEvaluableNode):
    """
    An object representing expression statements in the SLCO framework.

    Notes:
        - The ordering of variables within the expressions are not changed.
    """
    def __init__(self, op: str, values=None) -> None:
        super().__init__()
        self.op = operator_normalizations.get(op, op)

        # Note: Instead of left and right, the expression is depicted as an operation over an array of values.
        self._values = []
        if values is not None:
            self.values = values

    def __repr__(self) -> str:
        return f" {self.op} ".join(str(v) for v in self.values)

    def __iter__(self):
        """Iterate through all objects part of the AST structure."""
        for v in self.values:
            yield v

    def __eq__(self, o: object) -> bool:
        if isinstance(o, (Primary, Expression)):
            return self.is_equivalent(o)
        return False

    def __hash__(self) -> int:
        return 0

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, val) -> None:
        self._values[:] = val
        for v in self._values:
            v.parent = self


class Primary(SlcoStatementNode, SlcoEvaluableNode):
    """
    An object representing primary values in the SLCO framework.
    """
    def __init__(self, sign: str = "", value: Optional[Union[int, bool]] = None, target=None) -> None:
        super().__init__()
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

    def __repr__(self) -> str:
        if self.value is not None:
            exp_str = str(self.value).lower()
        elif self.ref is not None:
            exp_str = f"{self.ref}"
        else:
            exp_str = f"({self.body})"
        return f"!{exp_str}" if self.sign == "not" else self.sign + f"{exp_str}"

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


class VariableRef(SlcoStatementNode, SlcoEvaluableNode):
    """
    An object representing references to variables in the SLCO framework.

    Notes:
        - VariableRef objects do not necessarily need a parent. Hence, the parent can be ``None``.
    """
    def __init__(self, var: Variable, index=None) -> None:
        super().__init__()
        self.var = var
        self._index = None
        if index is not None:
            self.index = index

    def __repr__(self) -> str:
        var_str = self.var.name
        if self.index is not None:
            var_str += f"[{self.index}]"
        return var_str

    def __iter__(self):
        """Iterate through all objects part of the AST structure."""
        if self.index is not None:
            yield self.index

    def __eq__(self, o: object) -> bool:
        if isinstance(o, VariableRef):
            if self.var != o.var:
                return False
            if not self.var.is_array:
                return True

            try:
                # SMT is slow--optimize constants.
                i1 = int(str(self.index))
                i2 = int(str(o.index))
                return i1 == i2
            except ValueError:
                return self.is_equivalent(o)
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


class ActionRef(SlcoStatementNode):
    """
    An object representing references to model actions in the SLCO framework.
    """

    def __init__(self, act: Action) -> None:
        super().__init__()
        self.act: Action = act

    def __repr__(self) -> str:
        return str(self.act)

    def __iter__(self):
        yield self

    def __eq__(self, o: object) -> bool:
        if isinstance(o, ActionRef):
            return self.act == o.act
        return False

    def __hash__(self) -> int:
        return hash(self.act)


# ADDED CLASSES
class DecisionNode(SlcoLockableNode):
    """
    An object representing (non-)deterministic decision nodes in the code generator.
    """

    def __init__(
            self,
            is_deterministic: bool,
            decisions: List[Union[DecisionNode, Transition]],
            excluded_transitions: List[Transition]
    ):
        super().__init__()
        self.decisions = decisions
        self.is_deterministic = is_deterministic
        self._guard_statement = None

        # Track which decisions have been intentionally excluded in the decision node/structure.
        self.excluded_transitions = excluded_transitions
        for t in excluded_transitions:
            t.is_excluded = True

        # Track the variables used within the child nodes.
        self.used_variables: Set[Variable] = set()

        # Track the variables used within the child nodes that are location sensitive.
        self.location_sensitive_locks: List[Lock] = []

    def __repr__(self) -> str:
        return f"DecisionNode:{'DET' if self.is_deterministic else 'N_DET' if settings.use_random_pick else 'SEQ'}"

    @property
    def priority(self) -> int:
        return min(d.priority for d in self.decisions)

    @property
    def id(self) -> int:
        return min(d.id for d in self.decisions)

    @property
    def guard_statement(self):
        return self._guard_statement

    @guard_statement.setter
    def guard_statement(self, value):
        self._guard_statement = value

    def add_excluded_transitions(self, excluded_transitions):
        """Add transitions that are excluded by the decision node."""
        self.excluded_transitions += excluded_transitions
        for t in excluded_transitions:
            t.is_excluded = True

