from collections import OrderedDict
from typing import List, Dict, Set, Tuple, Union

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

    def render_lock_id_presence_contract_statements(self, model: SlcoLockableNode) -> str:
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

    def get_expression_control_node_success_closing_body(self, model: SlcoStatementNode) -> str:
        # Add a statement that triggers short-circuit evaluations.
        result = super().get_expression_control_node_success_closing_body(model)
        if isinstance(model, Expression) and model.op in ["or", "and"]:
            result = "\n".join(v.strip() for v in [result, "// Short-circuit fix trigger."] if v.strip() != "")
        return result

    def get_expression_control_node_contract_entries(self, model: SlcoStatementNode) -> Tuple[List[str], List[str]]:
        # Add the lock id presence check to the contract.
        result_statements, result_pure_functions = super().get_expression_control_node_contract_entries(model)
        statement = self.render_lock_id_presence_contract_statements(model)
        return result_statements + [statement], result_pure_functions

    def get_transition_contract_entries(self, model: Transition) -> Tuple[List[str], List[str]]:
        # Add the lock id presence check to the contract.
        result_statements, result_pure_functions = super().get_transition_contract_entries(model)
        statement = self.render_lock_id_presence_contract_statements(model.guard)
        return result_statements + [statement], result_pure_functions

    def render_decision_structure_lock_id_absence_contract_statements(self) -> str:
        """
        Render the VerCors statements that verify that the decision structure starts and ends with no locks active.
        """
        # Render the statement.
        return self.lock_id_absence_contract_template.render(
            target_locks_list_size=self.current_state_machine.target_locks_list_size
        )

    def get_decision_structure_contract_entries(self, model: StateMachine, state: State) -> Tuple[List[str], List[str]]:
        # Add the lock id presence check to the contract.
        result_statements, result_pure_functions = super().get_decision_structure_contract_entries(model, state)
        statement = self.render_decision_structure_lock_id_absence_contract_statements()
        return result_statements + [statement], result_pure_functions

    def render_state_machine_variable_declarations(self, model: StateMachine) -> str:
        # Add an instantiation for the lock_requests list.
        result = super().render_state_machine_variable_declarations(model)
        return self.join_with_strip([
            result,
            "// A list of lock requests. A value of 1 denotes that the given target is locked, and 0 implies no lock.",
            "private final int[] lock_requests;"
        ])

    def render_state_machine_constructor_body(self, model: StateMachine) -> str:
        # Add an instantiation for the lock_requests list.
        result = super().render_state_machine_constructor_body(model)
        return self.join_with_strip([
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

        # Link every lock to the associated variable reference.
        entry_node = model.locking_atomic_node.entry_node
        variable_ref_to_lock: Dict[VariableRef, Lock] = {i.original_ref: i for i in entry_node.original_locks}
        if len(variable_ref_to_lock) != len(entry_node.original_locks):
            raise Exception("Unexpected duplicate elements in original locks list.")

        # Gather all variables used in the statement, including non-class variables to add range assumptions.
        if isinstance(model, Assignment):
            variable_references: Set[VariableRef] = get_variable_references(model.left)
            if len(model.locking_atomic_node.child_atomic_nodes) == 0:
                variable_references.update(get_variable_references(model.right))
        else:
            variable_references: Set[VariableRef] = get_variable_references(model)

        # Create a dependency graph for the variable references, including both the class and local variables.
        ordered_references = self.get_topologically_ordered_variable_references(
            model, variable_references=variable_references
        )

        # Render assumptions based on the type of variable encountered.
        vercors_statements: List[str] = []
        for i in ordered_references:
            # Add a range assumption if the variable is an array.
            if i.var.is_array:
                # Assume that the index is within range.
                vercors_statements.append(f"//@ assume {self.render_range_check_statement(i)};")

            # Render a permission assumption if the reference is to a class variable.
            # Note that the function will error out if there is no lock for the target class variable--hence, only
            # variables associated to locks will get permissions assigned.
            if i.var.is_class_variable:
                # Find the associated lock object.
                lock: Lock = variable_ref_to_lock[i]

                if lock.unavoidable_location_conflict:
                    # Use the unpacked lock requests instead, since the original target is completely replaced.
                    for v in lock.lock_requests:
                        in_line_statement = self.get_plain_text_in_line_statement(v.ref)
                        vercors_statements.append(f"//@ assume Perm({in_line_statement}, 1); // Lock ids {v.id}")
                else:
                    # Render the original reference--the effect of rewrite rules will be verified in another step.
                    in_line_statement = self.get_plain_text_in_line_statement(i)
                    vercors_statements.append(
                        f"//@ assume Perm({in_line_statement}, 1);"
                    )

        # Remove duplicates but preserve order.
        vercors_statements = list(OrderedDict.fromkeys(vercors_statements))

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

    def get_statement_verification_method_contract_entries(
            self, model: SlcoStatementNode
    ) -> Tuple[List[str], List[str]]:
        """Get the statements that need to be included in the lock verification's contract."""
        # Use the statements included in the control node contract, since it is assumed to be identical.
        return self.get_expression_control_node_contract_entries(model)

    def get_statement_verification_method_contract(self, model: SlcoStatementNode) -> str:
        """Render the statement lock verification's contract."""
        target_statements, pure_functions = self.get_statement_verification_method_contract_entries(model)
        return self.render_method_contract(target_statements, pure_functions)

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

    def render_decision_structure(self, model: StateMachine, state: State) -> str:
        # The decision structure is not needed to verify the coverage of the locks.
        return ""

    def render_full_class_contract_permissions(self) -> str:
        # Override such that full variable permissions for class variables are no longer given.
        return ""

    def render_result_contract_statement(self, model: SlcoStatementNode, use_old_values: bool = False):
        # Override such that a result check is no longer performed.
        return ""

    def render_range_check_contract_statements(self, model: SlcoStatementNode, scope: str = "context") -> str:
        # Overwrite such that range checks are no longer performed.
        return ""

    def insert_range_check_assumption_method(self, model: SlcoStatementNode, transition_call: bool = False) -> str:
        # Overwrite such that range checks are no longer performed.
        return ""

    def render_class_contract_values_unaltered_statements(self) -> str:
        # Overwrite such that contract value change checks are no longer performed.
        return ""

    def render_state_machine_contract_values_unaltered_statements(self) -> str:
        # Overwrite such that contract value change checks are no longer performed.
        return ""


class VercorsLockingRewriteRulesModelRenderer(VercorsModelRenderer):
    """
    Create a subclass of the Vercors model renderer that renders VerCors verification statements to verify whether the
    rewrite rules applied to locks with array indices are applied correctly.
    """

    def __init__(self):
        super().__init__()

        # Overwrite the locking instructions template to instead work with ids and assumptions.
        self.transition_template = self.env.get_template(
            "vercors/locking/stripped_transition.jinja2template"
        )

        # Add additional templates.
        self.lock_rewrite_check_method_template = self.env.get_template(
            "vercors/locking/lock_rewrite_check_method.jinja2template"
        )

    def render_locking_instruction(self, model: LockingNode) -> str:
        # Disable the rendering of locking instructions, since the original program components will not be rendered.
        return ""

    def get_lock_rewrite_check_method_contract_entries(self) -> Tuple[List[str], List[str]]:
        """Get the contract statements that are needed the lock rewrite check methods."""
        return [
            self.render_common_contract_permissions(),
            self.render_base_state_machine_contract_permissions(),
            self.render_full_state_machine_contract_permissions(),
            self.render_base_class_contract_permissions(),
            self.render_full_class_contract_permissions()
        ], []

    def get_lock_rewrite_check_method_contract(self) -> str:
        """Get a permission contract for the lock rewrite check method."""
        # Pre-render data.
        target_statements, pure_functions = self.get_lock_rewrite_check_method_contract_entries()
        return self.render_method_contract(target_statements, pure_functions)

    def render_range_check_assumptions(self, model: SlcoStatementNode) -> List[str]:
        """Render the range check assumptions required to access the given variable reference as a list."""
        # Find all variables that need to be range checked.
        range_check_targets = self.get_range_check_variable_references(model)
        range_check_statements = [f"//@ assume {self.render_range_check_statement(s)};" for s in range_check_targets]
        return list(OrderedDict.fromkeys(range_check_statements))

    def add_lock_rewrite_check_method(self, model: Lock) -> None:
        """Add a method to the control node list that checks if the rewrite rules are applied correctly to the lock."""
        # The function will only be called for array variables.
        # Pre-render the appropriate data.
        lock_rewrite_check_method_contract = self.get_lock_rewrite_check_method_contract()
        method_name = f"{self.current_statement_prefix}_lock_rewrite_check_{len(self.current_control_node_methods)}"

        # Collect the statements that need to be rendered.
        rendered_statements = []

        # Save the value of the rewritten reference.
        rendered_statements.extend(self.render_range_check_assumptions(model.ref.index))
        rewritten_reference = self.get_plain_text_in_line_statement(model.ref.index)
        rendered_statements.append(f"//@ ghost int _index = {rewritten_reference}; // Lock {model}")

        # Render the rewrite rules.
        for target, value in model.rewrite_rules_list:
            # Make sure range assumptions are generated for the rewrite rules.
            rendered_statements.extend(self.render_range_check_assumptions(target))
            rendered_statements.extend(self.render_range_check_assumptions(value))
            target_statement = self.get_plain_text_in_line_statement(target)
            value_statement = self.get_plain_text_in_line_statement(value)
            rendered_statements.append(f"{target_statement} = {value_statement};")

        # Check if the rewritten and original values are equivalent.
        rendered_statements.extend(self.render_range_check_assumptions(model.ref.index))
        original_reference = self.get_plain_text_in_line_statement(model.original_ref.index)
        rendered_statements.append(f"//@ assert _index == {original_reference};")

        # Render the statement with the lock rewrite check method template and add it to the control nodes list.
        self.current_control_node_methods.append(
            self.lock_rewrite_check_method_template.render(
                lock_rewrite_check_method_contract=lock_rewrite_check_method_contract,
                method_name=method_name,
                rendered_statements=rendered_statements
            )
        )

    def render_locking_check(self, model: SlcoStatementNode) -> str:
        # Overwrite such that a new method is created for each original lock in which rewrite rules are verified.
        original_locks = model.locking_atomic_node.entry_node.original_locks
        for i in original_locks:
            var = i.original_ref.var
            if var.is_array and len(i.rewrite_rules_list) > 0:
                self.add_lock_rewrite_check_method(i)
        return ""

    def render_expression_control_node(
            self, model: Union[Expression, Primary, VariableRef], enforce_no_method_creation: bool = False
    ) -> str:
        # No longer render the control node methods, since the program structure is excluded in this test.
        # Remove the last entry in the supporting methods list if a method call is generated.
        result = super().render_expression_control_node(model, enforce_no_method_creation)
        if result.endswith("()"):
            self.current_control_node_methods.pop()
        return result

    def render_decision_structure(self, model: StateMachine, state: State) -> str:
        # The decision structure is not needed to verify the coverage of the locks.
        return ""

    def render_result_contract_statement(self, model: SlcoStatementNode, use_old_values: bool = False):
        # Override such that a result check is no longer performed.
        return ""

    def render_range_check_contract_statements(self, model: SlcoStatementNode, scope: str = "context") -> str:
        # Overwrite such that range checks are no longer performed.
        return ""

    def insert_range_check_assumption_method(self, model: SlcoStatementNode, transition_call: bool = False) -> str:
        # Overwrite such that range checks are no longer performed.
        return ""

    def render_class_contract_values_unaltered_statements(self) -> str:
        # Overwrite such that contract value change checks are no longer performed.
        return ""

    def render_state_machine_contract_values_unaltered_statements(self) -> str:
        # Overwrite such that contract value change checks are no longer performed.
        return ""
