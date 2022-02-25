from typing import List, Dict, Set, Tuple, Union

import networkx as nx

from objects.ast.interfaces import SlcoStatementNode, SlcoLockableNode
from objects.ast.models import Transition, StateMachine, State, Assignment, Expression, Primary, VariableRef
from objects.ast.util import get_variable_references
from objects.locking.models import LockingNode, Lock
from rendering.vercors.renderer import VercorsModelRenderer


class VercorsLockingStructureModelRenderer(VercorsModelRenderer):
    """
    Create a subclass of the Vercors model renderer that renders VerCors verification statements to verify the
    correctness of the rendered models from a locking structure point of view.
    """

    def __init__(self):
        super().__init__()

        # Overwrite the locking instructions template to instead work with ids and assumptions.
        self.locking_instruction_template = self.env.get_template(
            "vercors/locking/locking_instruction.jinja2template"
        )

        # Add additional templates.
        self.lock_id_presence_contract_template = self.env.get_template(
            "vercors/locking/lock_id_presence_contract.jinja2template"
        )
        self.lock_id_absence_contract_template = self.env.get_template(
            "vercors/locking/lock_id_absence_contract.jinja2template"
        )
        self.vercors_locking_check_template = self.env.get_template(
            "vercors/locking/locking_check.jinja2template"
        )

    def render_locking_check(self, model: SlcoStatementNode) -> str:
        # Use a different template and render the original locks instead.
        # Create a list containing all of the lock requests associated with the node.
        # Note that assignments do not need to be treated differently here, since the locking atomic node has been
        # constructed with the assignment differences already included.
        lock_requests = set()
        for i in model.locking_atomic_node.entry_node.original_locks:
            lock_requests.update(i.lock_requests)
        lock_requests = sorted(lock_requests, key=lambda x: x.id)

        lock_request_ids = [i.id for i in lock_requests]
        lock_request_comments = [self.get_variable_ref_comment(i.ref) for i in lock_requests]
        lock_request_entries = list(zip(lock_request_ids, lock_request_comments))

        if len(lock_requests) == 0:
            return ""
        else:
            return self.vercors_locking_check_template.render(
                lock_request_entries=lock_request_entries
            )

    def render_vercors_lock_id_presence_contract(self, model: SlcoLockableNode) -> str:
        """
        Render the VerCors statements that verify whether the locking structure is intact and behaving as expected.
        """
        # Find the lock requests present at the entry points and exit points of the given statement.
        entry_node_lock_requests = sorted(
            list(model.locking_atomic_node.entry_node.locking_instructions.requires_lock_requests), key=lambda x: x.id
        )
        success_exit_lock_requests = sorted(
            list(model.locking_atomic_node.success_exit.locking_instructions.ensures_lock_requests), key=lambda x: x.id
        )
        failure_exit_lock_requests = sorted(
            list(model.locking_atomic_node.failure_exit.locking_instructions.ensures_lock_requests), key=lambda x: x.id
        )

        # Create a disjunction of target ids for each of the nodes.
        entry_node_disjunction = " || ".join([f"_i == {i.id}" for i in entry_node_lock_requests])
        success_exit_disjunction = " || ".join([f"_i == {i.id}" for i in success_exit_lock_requests])
        failure_exit_disjunction = " || ".join([f"_i == {i.id}" for i in failure_exit_lock_requests])

        # Create human-readable comments for verification purposes.
        entry_node_comment_string = ", ".join([f"{i.id}: {i}" for i in entry_node_lock_requests])
        success_exit_comment_string = ", ".join([f"{i.id}: {i}" for i in success_exit_lock_requests])
        failure_exit_comment_string = ", ".join([f"{i.id}: {i}" for i in failure_exit_lock_requests])

        # Render the statement.
        return self.lock_id_presence_contract_template.render(
            entry_node_disjunction=entry_node_disjunction,
            success_exit_disjunction=success_exit_disjunction,
            failure_exit_disjunction=failure_exit_disjunction,
            entry_node_comment_string=entry_node_comment_string,
            success_exit_comment_string=success_exit_comment_string,
            failure_exit_comment_string=failure_exit_comment_string,
            target_locks_list_size=self.current_state_machine.target_locks_list_size
        )

    def render_vercors_expression_control_node_contract_requirements(
            self, model: SlcoStatementNode
    ) -> Tuple[str, List[str]]:
        # Add the lock request id check to the expression control node contract.
        requirements, pure_functions = super().render_vercors_expression_control_node_contract_requirements(model)
        lock_request_id_check = self.render_vercors_lock_id_presence_contract(model)
        return "\n\n".join(v.strip() for v in [requirements, lock_request_id_check] if v != ""), pure_functions

    def render_vercors_transition_contract_body(self, model: Transition) -> Tuple[str, List[str]]:
        # Add the lock request id check to the transition contract.
        result, pure_functions = super().render_vercors_transition_contract_body(model)
        lock_request_id_check = self.render_vercors_lock_id_presence_contract(model.guard)
        return "\n\n".join(v for v in [result, lock_request_id_check] if v != ""), pure_functions

    def render_vercors_decision_structure_lock_id_absence_contract(self) -> str:
        """
        Render the VerCors statements that verify that the decision structure starts and ends with no locks active.
        """
        # Render the statement.
        return self.lock_id_absence_contract_template.render(
            target_locks_list_size=self.current_state_machine.target_locks_list_size
        )

    def get_decision_structure_contract_body(self, model: StateMachine, state: State) -> Tuple[str, List[str]]:
        # Add the lock request id check to the decision node contract.
        result, pure_functions = super().get_decision_structure_contract_body(model, state)
        lock_request_id_check = self.render_vercors_decision_structure_lock_id_absence_contract()
        return "\n\n".join(v for v in [result, lock_request_id_check] if v != ""), pure_functions

    def render_state_machine_variable_declarations(self, model: StateMachine) -> str:
        # Add an instantiation for the lock_requests list.
        result = super().render_state_machine_variable_declarations(model)

        return "\n".join([
            result,
            "// A list of lock requests. A value of 1 denotes that the given target is locked, and 0 implies no lock.",
            "private final int[] lock_requests;"
        ])

    def render_state_machine_constructor_body(self, model: StateMachine) -> str:
        # Add an instantiation for the lock_requests list.
        result = super().render_state_machine_constructor_body(model)

        return "\n".join([
            result,
            "// Instantiate the lock requests array.",
            f"lock_requests = new int[{model.target_locks_list_size}];"
        ])


class VercorsLockingCoverageModelRenderer(VercorsModelRenderer):
    """
    Create a subclass of the Vercors model renderer that renders VerCors verification statements to verify whether the
    locks requested by statement are sufficient to execute it without permission errors.
    """

    def __init__(self):
        super().__init__()

        # Overwrite the transition template with a version that only renders the support methods.
        self.transition_template = self.env.get_template(
            "vercors/locking/stripped_transition.jinja2template"
        )

        # Add additional templates.
        self.statement_verification_method_template = self.env.get_template(
            "vercors/locking/statement_verification_method.jinja2template"
        )

    def render_locking_instruction(self, model: LockingNode) -> str:
        # Disable the rendering of locking instructions, since this test is done in isolation using the locks instead of
        # the lock requests--the presence of the lock has already been verified by the other model.
        return ""

    def render_locking_check(self, model: SlcoStatementNode) -> str:
        # Use a different template and render the original locks instead in the shape of assumptions.
        # Note that it is assumed that original locks are only created in leaf nodes--hence, throw an exception if this
        # assumption is violated. These tests are only to be performed in leaf nodes.
        atomic_node = model.locking_atomic_node
        entry_node = atomic_node.entry_node
        if not isinstance(model, Assignment) and len(atomic_node.child_atomic_nodes) > 0:
            # Nothing should be rendered for intermediate nodes. However, do check if the assumption is violated.
            if len(entry_node.original_locks) > 0:
                raise Exception(
                    "Assumption violated--non-empty original locks list in an atomic node that isn't a leaf node."
                )
            return ""

        # Gather all variables used in the statement, including non-class variables to add range assumptions.
        if isinstance(model, Assignment):
            variable_references: Set[VariableRef] = get_variable_references(model.left)
            if len(model.locking_atomic_node.child_atomic_nodes) == 0:
                variable_references.update(get_variable_references(model.right))
        else:
            variable_references: Set[VariableRef] = get_variable_references(model)

        # Link every lock to the associated variable reference.
        entry_node = model.locking_atomic_node.entry_node
        variable_ref_to_lock: Dict[VariableRef, Lock] = {i.original_ref: i for i in entry_node.original_locks}
        if len(variable_ref_to_lock) != len(entry_node.original_locks):
            raise Exception("Unexpected duplicate elements in original locks list.")

        # Create a dependency graph for the variable references, including both the class and local variables.
        graph = nx.DiGraph()
        while len(variable_references) > 0:
            source = variable_references.pop()
            graph.add_node(source)
            if source.index is not None:
                sub_references: Set[VariableRef] = get_variable_references(source.index)
                for target in sub_references:
                    # The edges are in reverse order, since inner variable references need to have priority.
                    graph.add_edge(target, source)

        # Create a topological ordering using the graph to avoid order sensitive permission errors.
        ordered_references: List[VariableRef] = list(nx.topological_sort(graph))

        # Render assumptions based on the type of variable encountered.
        vercors_statements: List[str] = []
        for i in ordered_references:
            # Add a range assumption if the variable is an array.
            if i.var.is_array:
                # Assume that the index is within range.
                index_in_line_statement = self.get_expression_control_node_in_line_statement(
                    i.index, enforce_no_method_creation=True
                )
                var = i.var
                var_name = f"c.{var.name}" if var.is_class_variable else f"{var.name}"
                vercors_statements.append(
                    f"//@ assume 0 <= {index_in_line_statement} && {index_in_line_statement} < {var_name}.length;"
                )

            # Render a permission assumption if the reference is to a class variable.
            if i.var.is_class_variable:
                # Find the associated lock object.
                lock: Lock = variable_ref_to_lock[i]

                if lock.unavoidable_location_conflict:
                    # Use the unpacked lock requests instead, since the original target is completely replaced.
                    for v in lock.lock_requests:
                        in_line_statement = self.get_expression_control_node_in_line_statement(
                            v.ref, enforce_no_method_creation=True
                        )
                        vercors_statements.append(
                            f"//@ assume Perm({in_line_statement}, 1); // Lock ids {v.id}"
                        )
                else:
                    # Render the original reference--the effect of rewrite rules will be verified in another step.
                    in_line_statement = self.get_expression_control_node_in_line_statement(
                        i, enforce_no_method_creation=True
                    )
                    vercors_statements.append(
                        f"//@ assume Perm({in_line_statement}, 1);"
                    )

        # Combine the statements with new lines.
        return "\n".join(vercors_statements)

    def get_expression_control_node_if_statement_failure_closing_body(self, model: SlcoStatementNode) -> str:
        # Do not include assertions, since the locks may not be acquired. Structural checks are done in p1.
        return ""

    def get_expression_control_node_success_closing_body(self, model: SlcoStatementNode) -> str:
        # Do not include assertions, since the locks may not be acquired. Structural checks are done in p1.
        return ""

    def get_expression_control_node_failure_closing_body(self, model: SlcoStatementNode) -> str:
        # Do not include assertions, since the locks may not be acquired. Structural checks are done in p1.
        return ""

    def render_range_assumptions(self, model: SlcoStatementNode) -> str:
        # The assumptions are instead generated by the locking check method.
        return ""

    def render_vercors_expression_control_node_contract_requirements(
            self, model: SlcoStatementNode
    ) -> Tuple[str, List[str]]:
        # Exclude the rendering of assumptions based on the control flow structure.
        return "", []

    def get_statement_verification_method_contract(self, model: SlcoStatementNode) -> str:
        """Render the statement lock verification's contract."""
        # Use the expression control node contract since it is assumed to be identical.
        return self.render_vercors_expression_control_node_contract(model)

    def include_statement_verification_method(
            self, model: SlcoStatementNode, statement_verification_method_body: str
    ) -> None:
        """Include a method for the given statement to verify the lock coverage in isolation per statement."""
        # Render the statement verification method template and include it as a control node method.
        statement_verification_method_contract = self.get_statement_verification_method_contract(model)
        statement_verification_method_name = self.current_statement_prefix

        self.current_control_node_methods.append(
            self.statement_verification_method_template.render(
                statement_verification_method_body=statement_verification_method_body,
                statement_verification_method_contract=statement_verification_method_contract,
                statement_verification_method_name=statement_verification_method_name
            )
        )

    def render_root_expression(self, model: Union[Expression, Primary]) -> str:
        result = super().render_root_expression(model)
        self.include_statement_verification_method(model, result)
        return result

    def render_assignment(self, model: Assignment) -> str:
        result = super().render_assignment(model)
        self.include_statement_verification_method(model, result)
        return result

    def render_transition(self, model: Transition) -> str:
        # Reset the current statement number.
        return super().render_transition(model)

    def render_decision_structure(self, model: StateMachine, state: State) -> str:
        # The decision structure is not needed to verify the coverage of the locks.
        return ""

    def render_vercors_contract_class_permissions(self) -> str:
        # Override the method to not include variable permissions for classes.
        class_array_variable_names = [v.name for v in self.current_class.variables if v.is_array]
        class_array_variable_lengths = [v.type.size for v in self.current_class.variables if v.is_array]
        class_array_variable_permissions = list(
            zip(class_array_variable_names, class_array_variable_lengths)
        )

        # Render the class permissions template.
        return self.class_permissions_template.render(
            class_array_variable_permissions=class_array_variable_permissions,
            class_variable_permissions=[]
        )
