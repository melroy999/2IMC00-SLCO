from collections import OrderedDict
from typing import List, Dict, Set, Tuple, Union

from objects.ast import util
from objects.ast.interfaces import SlcoStatementNode, SlcoLockableNode
from objects.ast.models import Transition, StateMachine, State, Assignment, Expression, Primary, VariableRef
from objects.ast.util import get_variable_references
from objects.locking.models import LockingNode, Lock, LockRequest
from preprocessing.ast.simplification import simplify
from rendering.vercors.renderer import VercorsModelRenderer


class VercorsLockingModelRenderer(VercorsModelRenderer):
    """
    Create a subclass of the Vercors model renderer that renders VerCors verification statements to verify the locking
    mechanism's integrity.
    """

    # TODO: Current plan:
    #   - Rewrite the required/ensured lists in the locking instructions to use localized rewritten versions.
    #       - (x) Or, rewrite the expressions to already have the assignments applied--this way, locks remain unchanged.
    #   - Find a way to check if any permissions remain at the end of the execution--there shouldn't be.

    # TODO: This part is also affected by the range assumption problem.
    #   - (x) Added a stronger range check generator that does include checks for local array variables.

    # FIXME: A primary concern at this moment is that the rewrite rules applied to the lock requests might not have the
    #  same effect as those applied to the assignments.
    #   - Applying the rewrite rules could result in variables disappearing.
    #   - Applying the rewrite rules could result in variables remaining active for longer.
    #   - It is not guaranteed that the lock requests will use the earliest assigned value--disparity can still exist.
    #   - This would result that this approach will always be faulty in more complex models, which is undesirable.

    # The issue mentioned above reveals an additional concern--can rewrites introduce additional variables that need to
    # be locked? Would this be handled appropriately?
    #   - [i != 0; x[i] := 0; i := k; x[i] := 0], suppose all are variables are class variables.
    #   - In this instance, the lock request for the second x[i] would become x[k]--would k still be locked?
    #   - The code does seem to handle this situation appropriately--even with a higher lock id for k, it is placed at
    #   the correct position.

    # TODO: The main conclusion is that this code is never going to work as desired. The original idea of having a list
    #  that holds all locked targets is still the most robust solution, but too slow to have any practical meaning.
    #   - This solution will work appropriately for models without at most one assignment per transition.

    def __init__(self):
        super().__init__()

        # Add support variables.
        self.current_locking_method_number = 0
        self.current_assignment_number = 0
        self.current_rewrite_rules: List[Tuple[VariableRef, Union[Expression, Primary]]] = []

        # Overwrite the assignment template, since they are applied as rewrite rules instead.
        self.assignment_template = self.env.get_template(
            "vercors/locking/revised/assignment.jinja2template"
        )
        # Add additional templates.
        self.class_variable_permissions_template = self.env.get_template(
            "vercors/locking/revised/class_variable_permissions.jinja2template"
        )
        self.locking_method_template = self.env.get_template(
            "vercors/locking/revised/locking_method.jinja2template"
        )
        self.locking_method_contract_body_template = self.env.get_template(
            "vercors/locking/revised/locking_method_contract_body.jinja2template"
        )
        self.assignment_method_template = self.env.get_template(
            "vercors/locking/revised/assignment_method.jinja2template"
        )
        self.assignment_method_contract_body_template = self.env.get_template(
            "vercors/locking/revised/assignment_method_contract_body.jinja2template"
        )

    def render_locking_check(self, model: SlcoStatementNode) -> str:
        # Locking checks are not needed for this particular model.
        return ""

    def get_locking_method_contract_body(self, model: LockingNode) -> str:
        """Get the locking method contract of the given node with all required permissions."""
        # Pre-render data.
        vercors_common_contract_permissions = self.render_vercors_contract_common_permissions()
        vercors_state_machine_contract_permissions = self.render_vercors_contract_state_machine_permissions()
        vercors_class_contract_permissions = self.render_vercors_contract_class_permissions()

        # Gather the contract class variable permission entries.
        requires_permissions_list = self.render_permission_list(model.locking_instructions.requires_lock_requests)
        ensures_permissions_list = self.render_permission_list(model.locking_instructions.ensures_lock_requests)

        # Render the vercors expression control node contract body template.
        return self.locking_method_contract_body_template.render(
            vercors_common_contract_permissions=vercors_common_contract_permissions,
            vercors_state_machine_contract_permissions=vercors_state_machine_contract_permissions,
            vercors_class_contract_permissions=vercors_class_contract_permissions,
            requires_permissions_list=requires_permissions_list,
            ensures_permissions_list=ensures_permissions_list
        )

    def insert_permission_range_check_statements(
            self, i: VariableRef, target_list: List[str], root_only: bool = False, operator: str = "&&"
    ) -> None:
        """
        Add statements to the given list that adds the appropriate range checks statements for the target statement.
        """
        if root_only:
            # Only add a range check for the root statement.
            if i.var.is_array:
                in_line_index_statement = self.get_expression_control_node_in_line_statement(
                    i.index, enforce_no_method_creation=True
                )
                target_list.append(
                    f"0 <= {in_line_index_statement} {operator} {in_line_index_statement} < {i.var.type.size}"
                )
        else:
            # Add the appropriate (potentially nested) range assumptions.
            range_check_targets = [v for v in self.get_topologically_ordered_variable_references({i}) if v.var.is_array]
            for target in range_check_targets:
                self.insert_permission_range_check_statements(target, target_list, root_only=True)

    def insert_permission_statements(
            self, i: VariableRef, target_list: List[str], write_permission: bool = False
    ) -> None:
        """Add statements to the given list that adds the appropriate range checks and permission statements."""
        # Add a permission statement for the main variable.
        in_line_statement = self.get_expression_control_node_in_line_statement(i, enforce_no_method_creation=True)
        if write_permission:
            target_list.append(f"Perm({in_line_statement}, 1)")
        else:
            target_list.append(f"Perm({in_line_statement}, 1\\{self.current_state_machine.target_locks_list_size})")

    def render_locking_method(self, model: LockingNode) -> str:
        """Insert a dummy method that simulates the locking mechanism through assumptions and permissions leaks."""
        # Generate a method that performs the locking instruction.
        locking_method_name = f"locking_operation_{self.current_locking_method_number}"
        locking_method_contract = self.get_locking_method_contract_body(model)

        # Create assumptions for each lock.
        assumption_targets = []
        for phase in model.locking_instructions.locks_to_acquire_phases:
            for i in phase:
                self.insert_permission_range_check_statements(i.ref, assumption_targets)
                self.insert_permission_statements(i.ref, assumption_targets)
        for i in model.locking_instructions.unpacked_lock_requests:
            self.insert_permission_range_check_statements(i.ref, assumption_targets)
            self.insert_permission_statements(i.ref, assumption_targets)

        # Render strings for lock releases to avoid confusion.
        lock_releases = [
            self.get_expression_control_node_in_line_statement(
                i.ref, enforce_no_method_creation=True
            ) for i in model.locking_instructions.locks_to_release
        ]

        # Add the locking method to the control nodes list.
        self.current_control_node_methods.append(
            self.locking_method_template.render(
                locking_method_contract=locking_method_contract,
                locking_method_name=locking_method_name,
                assumption_targets=assumption_targets,
                lock_releases=lock_releases
            )
        )

        # Increment the locking node counter.
        self.current_locking_method_number += 1

        # Return a call to the method.
        return f"{locking_method_name}();"

    def render_locking_instruction(self, model: LockingNode) -> str:
        # Create functions that attain the appropriate permissions--this will localize the effect of functions.
        if model.has_locks():
            return self.render_locking_method(model)
        else:
            return ""

    def render_permission_list(self, permissions: Set[LockRequest]) -> str:
        """Render the lock permissions list associated with the given list of permissions."""
        # Convert the lock requests to the appropriate variable references.
        variable_references = {i.ref for i in permissions}

        # Create a dependency graph for the variable references, including both the class and local variables.
        ordered_references = self.get_topologically_ordered_variable_references(variable_references)

        permission_targets = []
        for i in ordered_references:
            # Perform a bound check on the index of the target is an array.
            # Local variables are included too, since range checking is still necessary for these.
            self.insert_permission_range_check_statements(i, permission_targets, root_only=True, operator="**")

            # Only add a permission entry if the variable is a class variable.
            if i.var.is_class_variable:
                self.insert_permission_statements(i, permission_targets)

        return " ** ".join(permission_targets)

    def render_class_variable_permissions(self, model: SlcoStatementNode) -> str:
        """Render contract entries for variables that are expected to be locked at the entry and exit points."""
        # Find the permissions that need to be present at the start and end of the statement.
        required_permissions = model.locking_atomic_node.entry_node.locking_instructions.requires_lock_requests
        ensured_permissions_success = model.locking_atomic_node.success_exit.locking_instructions.ensures_lock_requests
        ensured_permissions_failure = model.locking_atomic_node.failure_exit.locking_instructions.ensures_lock_requests

        # Create lists of permissions joined with the operator "**".
        required_permissions_list = self.render_permission_list(required_permissions)
        ensured_permissions_success_list = self.render_permission_list(ensured_permissions_success)
        ensured_permissions_failure_list = self.render_permission_list(ensured_permissions_failure)

        # Render the class variable permissions template.
        return self.class_variable_permissions_template.render(
            required_permissions_list=required_permissions_list,
            ensured_permissions_success_list=ensured_permissions_success_list,
            ensured_permissions_failure_list=ensured_permissions_failure_list
        )

    def render_vercors_expression_control_node_contract_requirements(
            self, model: SlcoStatementNode
    ) -> Tuple[str, List[str]]:
        # Exclude all original requirements, since the required permissions are not present.
        permissions = self.render_class_variable_permissions(model).strip()
        return permissions, []

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

    def render_vercors_transition_contract_body(self, model: Transition) -> Tuple[str, List[str]]:
        # Add the lock request id check to the transition contract.
        result, pure_functions = super().render_vercors_transition_contract_body(model)
        permissions = self.render_class_variable_permissions(model.guard).strip()
        return "\n\n".join(v for v in [result, permissions] if v != ""), pure_functions

    @staticmethod
    def render_placeholder_string(model: SlcoStatementNode) -> str:
        """
        Render a string for conjunctions and disjunctions such that the fix for short-circuit evaluation works properly.
        """
        if isinstance(model, Expression) and model.op in ["or", "and"]:
            return "// Short-circuit evaluation trigger."
        else:
            return ""

    def get_expression_control_node_if_statement_failure_closing_body(self, model: SlcoStatementNode) -> str:
        # Do not include assertions, since the locks may not be acquired.
        return ""

    def get_expression_control_node_if_statement_success_closing_body(self, model: SlcoStatementNode) -> str:
        # Do not include assertions, since the locks may not be acquired.
        return ""

    def get_expression_control_node_success_closing_body(self, model: SlcoStatementNode) -> str:
        # Do not include assertions, since the locks may not be acquired.
        return self.render_placeholder_string(model)

    def get_expression_control_node_failure_closing_body(self, model: SlcoStatementNode) -> str:
        # Do not include assertions, since the locks may not be acquired.
        return self.render_placeholder_string(model)

    def render_range_assumptions(self, model: SlcoStatementNode, prefix="//@ assume") -> str:
        # Exclude, since the assumptions are part of the locking mechanism and contract.
        return ""

    def get_assignment_opening_body(self, model: Assignment) -> str:
        # Add a write assumption for the target variable.
        result = super().get_assignment_opening_body(model)

        # Perform the appropriate range checks.
        statements = []
        range_check_statements = []
        self.insert_permission_range_check_statements(model.left, range_check_statements)
        statements.extend([f"//@ assume {v};" for v in range_check_statements])

        # Also assert that permission is present over the element before elevation.
        permission_statements = []
        self.insert_permission_statements(model.left, permission_statements)
        self.insert_permission_statements(model.left, permission_statements, write_permission=True)
        statements.append(f"//@ assert {permission_statements[0]};")
        statements.append(f"//@ assume {permission_statements[1]};")

        return "\n".join(v for v in [result] + statements if v != "")

    def get_assignment_method_contract_body(self, model: Assignment) -> str:
        """Get the method contract of the given assignment with all required permissions."""
        # Pre-render data.
        vercors_common_contract_permissions = self.render_vercors_contract_common_permissions()
        vercors_state_machine_contract_permissions = self.render_vercors_contract_state_machine_permissions()
        vercors_class_contract_permissions = self.render_vercors_contract_class_permissions()

        # Gather the contract class variable permission entries.
        requires_permissions_list = self.render_permission_list(
            model.locking_atomic_node.entry_node.locking_instructions.requires_lock_requests
        )
        ensures_permissions_list = self.render_permission_list(
            model.locking_atomic_node.success_exit.locking_instructions.ensures_lock_requests
        )

        # Render the vercors expression control node contract body template.
        return self.assignment_method_contract_body_template.render(
            vercors_common_contract_permissions=vercors_common_contract_permissions,
            vercors_state_machine_contract_permissions=vercors_state_machine_contract_permissions,
            vercors_class_contract_permissions=vercors_class_contract_permissions,
            requires_permissions_list=requires_permissions_list,
            ensures_permissions_list=ensures_permissions_list
        )

    def render_assignment_method(self, model: Assignment) -> str:
        """Insert a wrapper method for a given assignment such that the write permission stays local."""
        # Generate a method that performs the locking instruction.
        assignment_method_name = f"assignment_{self.current_assignment_number}"
        assignment_method_contract = self.get_assignment_method_contract_body(model)
        assignment_method_body = super().render_assignment(model)

        self.current_control_node_methods.append(
            self.assignment_method_template.render(
                assignment_method_contract=assignment_method_contract,
                assignment_method_name=assignment_method_name,
                assignment_method_body=assignment_method_body
            )
        )

        # Increment the assignment node counter.
        self.current_assignment_number += 1

        # Return a call to the method.
        return f"{assignment_method_name}();"

    def render_assignment(self, model: Assignment) -> str:
        # Assignments are rewritten to no longer change the value of the target variable--this is done since it is too
        # difficult to add the appropriate verification data to the locking system.
        #   - Hence, instead of trying to restore the lock targets to pre-rewritten form, the assignment are applied as
        #   rewrite rules instead to all succeeding statements. This ensures that the lock requests and the statements
        #   that are being checked are over the same original variable values.
        #   - Note that this adds an additional point of failure in the verification process--this unfortunately cannot
        #   be avoided, due to time constraints and complexity issues encountered during the Thesis.

        # Generate the dictionary.
        rewrite_rules: Dict[VariableRef, Union[Expression, Primary]] = dict()
        for variable, replacement in self.current_rewrite_rules:
            target_variable = variable
            if variable.var.is_array:
                # Rewrite target_variable appropriately if it has an index.
                target_variable = util.copy_node(variable, dict(), dict())
                target_variable.index = util.copy_node(target_variable.index, dict(), rewrite_rules)

            # Apply the rewrite rule to the replacement statement and note down the change of value.
            rewrite_rules[target_variable] = util.copy_node(replacement, dict(), rewrite_rules)

        rewritten_model = util.copy_node(model, dict(), dict())
        rewritten_model.right = simplify(util.copy_node(model.right, dict(), rewrite_rules))
        if model.left.var.is_array:
            rewritten_model.left.index = simplify(util.copy_node(model.left.index, dict(), rewrite_rules))

        # Isolate the assignment such that the write assumption only holds for a limited scope.
        method_call = self.render_assignment_method(rewritten_model)

        # Add the assignment as a rewrite rule.
        self.current_rewrite_rules.append((model.left, model.right))

        return method_call

    def render_transition(self, model: Transition) -> str:
        # Reset the current rewrite rules.
        self.current_rewrite_rules: List[Tuple[VariableRef, Union[Expression, Primary]]] = []
        return super().render_transition(model)


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

