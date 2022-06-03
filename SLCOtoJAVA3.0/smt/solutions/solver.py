from typing import List, Dict, Callable, Optional, Union, Tuple

from z3 import z3, ModelRef, CheckSatResult

from objects.ast.models import Transition, DecisionNode


class DecisionStructureSolver:
    """A class that converts a list of transitions to a decision structure."""

    def __init__(self, transitions: List[Transition]) -> None:
        super().__init__()

        # Give all transitions an unique identity.
        for i, t in enumerate(transitions):
            t.id = i

        # The transitions the decision is made over.
        self.transitions = list(transitions)

        # The Z3 SMT solver object.
        self.solver = z3.Optimize()

        # The truth tables.
        self.and_truth_table: Optional[Dict[int, Dict[int, bool]]] = None
        self.is_equal_truth_table: Optional[Dict[int, Dict[int, bool]]] = None
        self.contains_truth_table: Optional[Dict[int, Dict[int, bool]]] = None

    def create_truth_table(
            self,
            transitions: List[Transition],
            alias_variables: Dict[str, z3.ArithRef],
            prefix: str,
            target_expression: Callable[[object, object], object],
            mirrored=False,
            target_value=z3.Z3_L_FALSE
    ) -> Dict[int, Dict[int, bool]]:
        """Create a truth table for the given target expression."""
        # Create the truth table and the associated variables.
        truth_table: Dict[int, Dict[int, bool]] = dict()
        for t1 in transitions:
            truth_table[t1.id]: Dict[int, bool] = {t2.id: False for t2 in transitions}
            for t2 in transitions:
                alias_variables[f"{prefix}{t1.id}_{t2.id}"] = z3.Bool(f"{prefix}{t1.id}_{t2.id}")

        # Insert the appropriate values.
        for t1 in transitions:
            for t2 in transitions:
                if mirrored and t2.id > t1.id:
                    break

                # Add a new AND relation to the solver.
                e1 = t1.guard.smt
                e2 = t2.guard.smt

                # Perform the check. Existential quantifiers are not supported, so do the check separately.
                self.solver.push()
                self.solver.add(target_expression(e1, e2))
                # noinspection DuplicatedCode
                result = self.solver.check().r == target_value
                self.solver.pop()

                # Add the constant to the solver.
                self.solver.add(alias_variables[f"{prefix}{t1.id}_{t2.id}"] == result)
                truth_table[t1.id][t2.id] = result
                if mirrored and t2.id < t1.id:
                    self.solver.add(
                        alias_variables[f"{prefix}{t2.id}_{t1.id}"] == alias_variables[f"{prefix}{t1.id}_{t2.id}"]
                    )
                    truth_table[t2.id][t1.id] = result

        # Return the truth table.
        return truth_table

    def create_and_truth_table(
            self,
            transitions: List[Transition],
            alias_variables: Dict[str, z3.ArithRef],
    ) -> Dict[int, Dict[int, bool]]:
        """Create a truth table for the AND operation between the given list of transitions."""
        return self.create_truth_table(
            transitions, alias_variables, "and", lambda e1, e2: z3.And(e1, e2), mirrored=True, target_value=z3.Z3_L_TRUE
        )

    def create_is_equal_truth_table(
            self,
            transitions: List[Transition],
            alias_variables: Dict[str, z3.ArithRef],
    ) -> Dict[int, Dict[int, bool]]:
        """Create a truth table for the negated equality operation between the given list of transitions."""
        return self.create_truth_table(
            transitions, alias_variables, "ieq", lambda e1, e2: z3.Not(e1 == e2), mirrored=True
        )

    def create_contains_truth_table(
            self,
            transitions: List[Transition],
            alias_variables: Dict[str, z3.ArithRef],
    ) -> Dict[int, Dict[int, bool]]:
        """Create a truth table for the subset relations between transitions, being true if one contains the other."""
        return self.create_truth_table(
            transitions, alias_variables, "contains", lambda e1, e2: z3.Not(z3.Implies(e2, e1))
        )

    @staticmethod
    def get_group_variable_bounds(v: z3.ArithRef, transitions: List[Transition]) -> z3.ArithRef:
        """Get the bounds to use for the group variable within the smt model."""
        return z3.And(v >= 0, v < 2)

    @staticmethod
    def include_contains_truth_table() -> bool:
        """A boolean that denotes whether truth table variables for the contains operator needs to be included."""
        return False

    def create_smt_model_variables(
            self, transitions: List[Transition]
    ) -> Dict[str, z3.ArithRef]:
        """Create a dictionary of SMT variables and apply the appropriate bounds."""
        alias_variables: Dict[str, z3.ArithRef] = dict()

        # Create variables for each transition that will indicate the number of the group they are in.
        for t in transitions:
            v = alias_variables[f"g{t.id}"] = z3.Int(f"g{t.id}")
            self.solver.add(self.get_group_variable_bounds(v, transitions))

        # Create support variables for the priorities assigned to the transitions.
        for t in transitions:
            v = alias_variables[f"p{t.id}"] = z3.Int(f"p{t.id}")
            self.solver.add(v == t.priority)

        # Create the desired truth tables.
        self.and_truth_table = self.create_and_truth_table(transitions, alias_variables)
        self.is_equal_truth_table = self.create_is_equal_truth_table(transitions, alias_variables)
        if self.include_contains_truth_table():
            self.contains_truth_table = self.create_contains_truth_table(transitions, alias_variables)

        return alias_variables

    def solve_smt_model(self) -> Tuple[CheckSatResult, ModelRef]:
        """Get a solution for the current SMT model."""
        result, model = self.solver.check(), self.solver.model()
        if result.r == z3.Z3_L_UNDEF:
            print(result, model)
            raise Exception("Unknown result.")
        if result.r == z3.Z3_L_FALSE:
            print(result, model)
            raise Exception("Unsatisfiable result.")
        return result, model

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def create_deterministic_node(
            self, transitions: List[Transition], alias_variables: Dict[str, z3.ArithRef]
    ) -> Union[DecisionNode, Transition]:
        """Select non-deterministic groups within the given list of transitions."""
        # Sort the transitions and create a deterministic node.
        transitions.sort(key=lambda x: (x.priority, x.id))
        deterministic_node = DecisionNode(True, transitions, [])
        return deterministic_node

    def remove_conflicting_transitions(self, transitions: List[Transition]):
        """Remove transitions from the transition list that overlap with every other transition."""
        conflicting_transitions = []
        for t1 in transitions:
            if all(self.and_truth_table[t1.id][t2.id] for t2 in transitions):
                conflicting_transitions.append(t1)
        for t in conflicting_transitions:
            transitions.remove(t)

        # Return the list of conflicting transitions.
        return conflicting_transitions

    def create_non_deterministic_node_smt_model_overlap_constraints(
            self,
            t1: Transition,
            v: z3.ArithRef,
            transitions: List[Transition],
            alias_variables: Dict[str, z3.ArithRef],
            target_group: int = 1
    ) -> None:
        """Add the overlap constraints to the model."""
        inner_or = z3.Or([
            z3.And(
                alias_variables[f"g{t2.id}"] == target_group,
                alias_variables[f"and{t1.id}_{t2.id}"]
            ) for t2 in transitions if t1.id != t2.id
        ])
        self.solver.add(z3.Implies(v == target_group, z3.Not(inner_or)))

    def create_non_deterministic_node_smt_model_priority_constraints(
            self,
            t1: Transition,
            v: z3.ArithRef,
            transitions: List[Transition],
            alias_variables: Dict[str, z3.ArithRef],
            target_group: int = 1
    ) -> None:
        """Add the priority constraints to the model."""
        inner_or = z3.Or([
            z3.And(
                alias_variables[f"g{t2.id}"] == target_group,
                alias_variables[f"p{t2.id}"] != alias_variables[f"p{t1.id}"]
            ) for t2 in transitions if t1.id != t2.id
        ])
        self.solver.add(z3.Implies(v == target_group, z3.Not(inner_or)))

    def create_non_deterministic_node_smt_model_optimization_constraint(
            self, transitions: List[Transition], alias_variables: Dict[str, z3.ArithRef]
    ) -> None:
        """Add the maximization constraints to the model."""
        self.solver.maximize(sum(alias_variables[f"g{t.id}"] for t in transitions))

    def create_non_deterministic_node_smt_model_constraints(
            self, transitions: List[Transition], alias_variables: Dict[str, z3.ArithRef]
    ) -> None:
        """Add the constraints needed to extract groups within non-deterministic nodes."""
        # Create the rules needed to construct valid groupings.
        for t1 in transitions:
            v = alias_variables[f"g{t1.id}"]

            # It must hold that the transition has no overlap with any of the members in the same group.
            self.create_non_deterministic_node_smt_model_overlap_constraints(t1, v, transitions, alias_variables)

            # It must hold that the priority of each member in the group is the same.
            self.create_non_deterministic_node_smt_model_priority_constraints(t1, v, transitions, alias_variables)

        # Maximize the size of the group.
        self.create_non_deterministic_node_smt_model_optimization_constraint(transitions, alias_variables)

    def get_non_deterministic_node_groups(
            self, transitions: List[Transition], alias_variables: Dict[str, z3.ArithRef]
    ) -> List[List[Union[DecisionNode, Transition]]]:
        """Convert the list of transitions to groups to be contained within a non-deterministic decision node."""
        # Assign the given transitions to groups that internally display deterministic behavior.
        deterministic_transition_groups: List[List[Union[DecisionNode, Transition]]] = []

        # Select groups until no transitions remain.
        while len(transitions) > 0:
            # Save the current state of the solver.
            self.solver.push()

            # Introduce the appropriate model constraints for the group selection.
            self.create_non_deterministic_node_smt_model_constraints(transitions, alias_variables)

            # Get the model solution.
            result, model = self.solve_smt_model()

            # Find the target transitions that are part of the deterministic group.
            grouped_transitions: List[Transition] = [
                t for t in transitions if model.evaluate(alias_variables[f"g{t.id}"], model_completion=True) == 1
            ]
            deterministic_transition_groups.append(grouped_transitions)

            # Restore the state of the solver.
            self.solver.pop()

            # Remove the transitions from the to be processed list.
            for t in grouped_transitions:
                transitions.remove(t)

        # Return the gathered groups.
        return deterministic_transition_groups

    def create_non_deterministic_node(
            self, transitions: List[Transition], alias_variables: Dict[str, z3.ArithRef]
    ) -> DecisionNode:
        """Convert the list of transitions to a non-deterministic decision node."""
        # Preprocess transitions that overlap with all other available transitions.
        conflicting_transitions = self.remove_conflicting_transitions(transitions)

        # Assign the remaining transitions to groups that internally display deterministic behavior.
        deterministic_transition_groups: List[
            List[Union[DecisionNode, Transition]]] = self.get_non_deterministic_node_groups(
            transitions, alias_variables
        )

        # Process the selected groups.
        non_deterministic_choices: List[Union[DecisionNode, Transition]] = []
        for g in deterministic_transition_groups:
            # Find remaining non-deterministic behavior within the deterministic group.
            deterministic_node = self.create_deterministic_node(g, alias_variables)

            # Add a deterministic decision node to the list of non-deterministic choices.
            non_deterministic_choices.append(deterministic_node)

        # Sort the non-deterministic choices and create a non-deterministic node.
        non_deterministic_choices += conflicting_transitions
        non_deterministic_choices.sort(key=lambda x: (x.priority, x.id))
        non_deterministic_node = DecisionNode(False, non_deterministic_choices, [])
        return non_deterministic_node

    def simplify(self, d: DecisionNode) -> DecisionNode:
        """Recursively simplify the structure of the given decision node."""
        excluded_transitions = d.excluded_transitions
        decisions = []
        guard_statement = d.guard_statement

        # Simplify decisions that contain only one option.
        for decision in d.decisions:
            if isinstance(decision, Transition):
                decisions.append(decision)
            else:
                # Simplify the nested transition.
                simplified_decision_node = self.simplify(decision)

                # Raise the decision if the decision node only contains one option.
                if len(simplified_decision_node.decisions) == 1:
                    decisions += simplified_decision_node.decisions
                    excluded_transitions += simplified_decision_node.excluded_transitions
                else:
                    decisions.append(simplified_decision_node)

        simplified_decision_node = DecisionNode(d.is_deterministic, decisions, excluded_transitions)
        simplified_decision_node.guard_statement = guard_statement
        return simplified_decision_node

    def validate(self, d: DecisionNode) -> None:
        """Validate the given decision structure for semantic correctness."""
        # Validate that all options are mutually exclusive if the node is deterministic.
        if d.is_deterministic:
            for i, d1 in enumerate(d.decisions):
                for j, d2 in enumerate(d.decisions):
                    # Overlap is mirrored. So break on observing equal decisions.
                    if i == j:
                        break

                    # Raise an exception if they have solutions in common.
                    id1 = d1.id if isinstance(d1, Transition) else d1.guard_statement.id
                    id2 = d2.id if isinstance(d2, Transition) else d2.guard_statement.id
                    if self.and_truth_table[id1][id2]:
                        raise Exception("The deterministic group has overlapping solution spaces.")

        # Validate that all excluded transitions are unreachable.
        for transition in d.excluded_transitions:
            if not transition.guard.is_false():
                raise Exception("A transition is excluded that can be active.")

        for decision in d.decisions:
            # Validate the nested decision nodes.
            if isinstance(decision, DecisionNode):
                self.validate(decision)

    def solve(self) -> DecisionNode:
        """Convert the list of transitions to a decision structure."""
        # Filter out transitions that are always true or false.
        false_transitions = []
        remaining_transitions = []
        for t in self.transitions:
            if t.guard.is_false():
                false_transitions.append(t)
            else:
                remaining_transitions.append(t)

        # Save the current state of the solver.
        self.solver.push()

        # Create all the variables needed within the SMT solution.
        alias_variables = self.create_smt_model_variables(remaining_transitions)

        # Select a non-deterministic root node, and re-create it to add the true and false transitions.
        non_deterministic_node = self.create_non_deterministic_node(remaining_transitions, alias_variables)
        non_deterministic_node.add_excluded_transitions(false_transitions)

        # Restore the state of the solver.
        self.solver.pop()

        # TODO: do sorting based on lock ordering optimization.

        # Post-process the node.
        decision_node = self.simplify(non_deterministic_node)

        # TODO: mark unavoidable lock violations due to the ordering.

        # Validate whether the deterministic nodes have non-overlapping members.
        self.validate(decision_node)

        # Return the node.
        return decision_node
