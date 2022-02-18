from __future__ import annotations

from typing import List, Tuple, Optional, Union, Callable, Dict, Set

import jinja2

import settings
from objects.ast.interfaces import SlcoStatementNode
from objects.ast.models import StateMachine, Class, SlcoModel, Variable, Object, State, Transition, DecisionNode, \
    Expression, Assignment, Composite, Primary, VariableRef
from objects.ast.util import get_variables_to_be_locked
from objects.locking.models import LockingNode


class JavaModelRenderer:
    """
    A renderer class that has a dual purpose: the class renders a given SLCO model as Java code.

    More crucially, the renderer class allows for certain functionality to be replaced for verification purposes, while
    also ensuring to a certain degree that the code remains maintainable and traceable through inheritance.
    """

    def __init__(self):
        # Create a jinja2 rendering environment.
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("jinja2_templates"),
            extensions=["jinja2.ext.loopcontrols", "jinja2.ext.do"],
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Create support variables to track intermediate data.
        self.current_control_node_id: int = 0
        self.current_statement_prefix: str = ""
        self.current_transition_prefix: str = ""
        self.current_control_node_methods: List[str] = []
        self.current_transition: Optional[Transition] = None
        self.current_state_machine: Optional[StateMachine] = None
        self.current_class: Optional[Class] = None

        # Lookup tables.
        self.operator_conversion_mapping: Dict[str, str] = {
            "<>": "!=",
            "=": "==",
            "and": "&&",
            "or": "||",
            "not": "!"
        }
        self.operator_render_function: Dict[str, Callable[[Expression, bool], str]] = {
            "**": self.render_power_operation,
            "%": self.render_modulo_operation
        }

        # Register the filters and templates.
        self.expression_template = self.env.get_template("java/statements/expression.jinja2template")
        self.assignment_template = self.env.get_template("java/statements/assignment.jinja2template")
        self.composite_template = self.env.get_template("java/statements/composite.jinja2template")
        self.transition_template = self.env.get_template("java/transition.jinja2template")
        self.state_machine_template = self.env.get_template("java/state_machine.jinja2template")
        self.class_template = self.env.get_template("java/class.jinja2template")
        self.object_instantiation = self.env.get_template("java/util/object_instantiation.jinja2template")
        self.lock_manager_template = self.env.get_template("java/locking/lock_manager.jinja2template")
        self.model_template = self.env.get_template("java/model.jinja2template")

        self.expression_control_node_if_statement_template = self.env.get_template(
            "java/statements/expression_control_node_if_statement.jinja2template"
        )
        self.expression_control_node_body_template = self.env.get_template(
            "java/statements/expression_control_node_body.jinja2template"
        )
        self.expression_control_node_template = self.env.get_template(
            "java/statements/expression_control_node.jinja2template"
        )
        self.transition_call_template = self.env.get_template(
            "java/decision_structures/transition_call.jinja2template"
        )
        self.deterministic_decision_node_template = self.env.get_template(
            "java/decision_structures/deterministic_decision_node.jinja2template"
        )
        self.pick_random_decision_node_template = self.env.get_template(
            "java/decision_structures/pick_random_decision_node.jinja2template"
        )
        self.sequential_decision_node_template = self.env.get_template(
            "java/decision_structures/sequential_decision_node.jinja2template"
        )
        self.decision_structure_template = self.env.get_template(
            "java/decision_structures/decision_structure.jinja2template"
        )

        self.locking_check_template = self.env.get_template("java/locking/locking_check.jinja2template")
        self.locking_instruction_template = self.env.get_template("java/locking/locking_instruction.jinja2template")

    def get_variable_ref_comment(self, model: VariableRef) -> str:
        """Create an easily identifiable comment for variable reference statements."""
        # Create an easily identifiable comment.
        return self.render_expression_control_node(model, enforce_no_method_creation=True)

    def render_locking_target(self, model: VariableRef) -> str:
        """Get the locking target (id + offset) of the given variable ref."""
        target = model.var.lock_id
        if model.index is not None:
            target = f"{target} + {self.render_expression_control_node(model.index, enforce_no_method_creation=True)}"
        return str(target)

    def render_locking_instruction(self, model: LockingNode) -> str:
        """Render the lock instruction within the given locking node as Java code."""
        if not model.has_locks():
            # Simply return an empty string if no locks need to be rendered.
            return ""

        # The target instruction.
        unpacked_lock_requests = model.locking_instructions.unpacked_lock_requests
        locks_to_release = model.locking_instructions.locks_to_release

        # Pre-render applicable information and data.
        locks_to_acquire_phases = []
        for locks_to_acquire in model.locking_instructions.locks_to_acquire_phases:
            locks_to_acquire_ids = [i.id for i in locks_to_acquire]
            locks_to_acquire_targets = [self.render_locking_target(i.ref) for i in locks_to_acquire]
            locks_to_acquire_comments = [self.get_variable_ref_comment(i.ref) for i in locks_to_acquire]
            locks_to_acquire_entries = list(
                zip(locks_to_acquire_ids, locks_to_acquire_targets, locks_to_acquire_comments)
            )
            locks_to_acquire_entries.sort()
            locks_to_acquire_phases.append(locks_to_acquire_entries)

        unpacked_lock_requests_ids = [i.id for i in unpacked_lock_requests]
        unpacked_lock_requests_targets = [self.render_locking_target(i.ref) for i in unpacked_lock_requests]
        unpacked_lock_requests_comments = [self.get_variable_ref_comment(i.ref) for i in unpacked_lock_requests]
        unpacked_lock_requests_entries = list(
            zip(unpacked_lock_requests_ids, unpacked_lock_requests_targets, unpacked_lock_requests_comments)
        )
        unpacked_lock_requests_entries.sort()

        locks_to_release_ids = [i.id for i in locks_to_release]
        locks_to_release_comments = [self.get_variable_ref_comment(i.ref) for i in locks_to_release]
        locks_to_release_entries = list(
            zip(locks_to_release_ids, locks_to_release_comments)
        )
        locks_to_release_entries.sort()

        # Render the locking instruction template.
        return self.locking_instruction_template.render(
            locks_to_acquire_phases=locks_to_acquire_phases,
            unpacked_lock_requests_entries=unpacked_lock_requests_entries,
            locks_to_release_entries=locks_to_release_entries,
        )

    def render_locking_check(self, model: SlcoStatementNode) -> str:
        """Render code that checks if all of the locks required by the given statement have been acquired."""
        if not settings.verify_locks:
            # Only render the locking check if requested.
            return ""

        # Find the locks used by the target statement.
        if isinstance(model, Assignment):
            target_variable_references: Set[VariableRef] = get_variables_to_be_locked(model.left)
            if len(model.locking_atomic_node.child_atomic_nodes) == 0:
                target_variable_references.update(get_variables_to_be_locked(model.right))
        elif len(model.locking_atomic_node.child_atomic_nodes) > 0:
            # TODO: Certain configurations may have intermediate control nodes too, for which the locks are not present.
            #   - Say, a && b becomes a control node, then a and b will not be locked prior to reaching a && b.
            #   - Note: currently, this is resolved by needing the lock to have only a single child node.
            target_variable_references: Set[VariableRef] = set()
        else:
            # Find which variables are used by the object in question.
            target_variable_references: Set[VariableRef] = get_variables_to_be_locked(model)

        # Convert the set to a list to have a constant ordering.
        target_lock_checks = list(target_variable_references)

        # Pre-render applicable information and data.
        locking_check_targets = [self.render_locking_target(i) for i in target_lock_checks]
        locking_check_comments = [self.get_variable_ref_comment(i) for i in target_lock_checks]
        locking_check_entries = list(
            zip(locking_check_targets, locking_check_comments)
        )
        locking_check_entries.sort()

        # Render the locking check template.
        return self.locking_check_template.render(
            locking_check_entries=locking_check_entries
        )

    def render_power_operation(self, model: Expression, enforce_no_method_creation: bool) -> str:
        """Render the given power operation as in-line Java code."""
        if len(model.values) != 2:
            raise Exception("The power operation needs to be between two values.")
        left_str = self.render_expression_control_node(model.values[0], enforce_no_method_creation)
        right_str = self.render_expression_control_node(model.values[1], enforce_no_method_creation)
        return f"(int) Math.pow({left_str}, {right_str})"

    def render_modulo_operation(self, model: Expression, enforce_no_method_creation: bool) -> str:
        """Render the given power operation as in-line Java code."""
        if len(model.values) != 2:
            raise Exception("The modulo operation needs to be between two values.")
        left_str = self.render_expression_control_node(model.values[0], enforce_no_method_creation)
        right_str = self.render_expression_control_node(model.values[1], enforce_no_method_creation)
        return f"Math.floorMod({left_str}, {right_str})"

    def render_operation_default(self, model: Expression, enforce_no_method_creation: bool) -> str:
        """Default method to render the given operation as in-line Java code."""
        values_str = [self.render_expression_control_node(v, enforce_no_method_creation) for v in model.values]
        return f" {self.operator_conversion_mapping.get(model.op, model.op)} ".join(values_str)

    def render_expression(self, model: Expression, enforce_no_method_creation: bool) -> str:
        """Render the given expression object as in-line Java code."""
        render_function = self.operator_render_function.get(model.op, self.render_operation_default)
        return render_function(model, enforce_no_method_creation)

    def render_primary(self, model: Primary, enforce_no_method_creation: bool) -> str:
        """Render the given primary object as in-line Java code."""
        if model.value is not None:
            exp_str = str(model.value).lower()
        elif model.ref is not None:
            exp_str = self.render_expression_control_node(model.ref, enforce_no_method_creation)
        else:
            exp_str = "(%s)" % self.render_expression_control_node(model.body, enforce_no_method_creation)
        return ("!(%s)" if model.sign == "not" else model.sign + "%s") % exp_str

    def render_variable_ref(self, model: VariableRef) -> str:
        """Render the given variable reference object as in-line Java code."""
        result = model.var.name
        if model.index is not None:
            result += "[%s]" % self.render_expression_control_node(model.index, enforce_no_method_creation=True)
        return result

    def get_control_node_name(self) -> str:
        """Get an unique name for the control node."""
        return f"{self.current_statement_prefix}_n_{len(self.current_control_node_methods)}"

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_if_statement_opening_body(self, model: SlcoStatementNode) -> str:
        """Get the opening statements of the expression control node object."""
        return ""

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_if_statement_success_closing_body(self, model: SlcoStatementNode) -> str:
        """Get the closing statements of the expression control node object's success branch."""
        return ""

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_if_statement_failure_closing_body(self, model: SlcoStatementNode) -> str:
        """Get the closing statements of the expression control node object's failure branch."""
        return ""

    @staticmethod
    def get_expression_control_node_comment(model: SlcoStatementNode):
        """Create an easily identifiable comment for expression wrapper methods."""
        # Create an easily identifiable comment.
        statement_comment = f"// SLCO expression wrapper | {model}"
        return statement_comment

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_contract(self, model: SlcoStatementNode) -> str:
        """Get the contract of the expression control node method."""
        return ""

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_opening_body(self, model: SlcoStatementNode) -> str:
        """Get the opening statements of the expression control node object."""
        entry_locking_instruction = self.render_locking_instruction(model.locking_atomic_node.entry_node).strip()
        locking_check = self.render_locking_check(model)
        return "\n".join(v for v in [entry_locking_instruction, locking_check] if v != "")

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_success_closing_body(self, model: SlcoStatementNode) -> str:
        """Get the closing statements of the expression control node object's success branch."""
        success_exit_locking_instruction = self.render_locking_instruction(model.locking_atomic_node.success_exit)
        return success_exit_locking_instruction

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_failure_closing_body(self, model: SlcoStatementNode) -> str:
        """Get the closing statements of the expression control node object's failure branch."""
        failure_exit_locking_instruction = self.render_locking_instruction(model.locking_atomic_node.failure_exit)
        return failure_exit_locking_instruction

    @staticmethod
    def requires_atomic_node_method(model: SlcoStatementNode) -> bool:
        """Check whether the given statement needs a control node method rendered."""
        return model.locking_atomic_node is not None and (model.locking_atomic_node.has_locks() or (
                settings.verify_locks and
                len(model.locking_atomic_node.child_atomic_nodes) == 0 and
                len(model.locking_atomic_node.entry_node.original_locks) > 0
            )
        )

    def get_expression_control_node_in_line_statement(
            self, model: SlcoStatementNode, enforce_no_method_creation: bool
    ) -> str:
        """Render the in-line statement of the expression control node."""
        # Construct the in-line conditional.
        if isinstance(model, Expression):
            in_line_statement = self.render_expression(model, enforce_no_method_creation)
        elif isinstance(model, Primary):
            in_line_statement = self.render_primary(model, enforce_no_method_creation)
        elif isinstance(model, VariableRef):
            in_line_statement = self.render_variable_ref(model)
        else:
            raise Exception(f"No function exists to turn objects of type {type(model)} into in-line Java statements.")
        return in_line_statement

    def render_expression_control_node_if_statement(
            self,
            model: Union[Expression, Primary, VariableRef],
            enforce_no_method_creation: bool,
            expression_control_node_success_closing_body: str = "",
            nested_statement: str = "",
            in_line_statement: str = None
    ) -> str:
        """Render the if statement within the expression control node."""
        # Get the in-line statement or use the already existing one if given.
        if in_line_statement is None:
            in_line_statement = self.get_expression_control_node_in_line_statement(model, enforce_no_method_creation)

        # Pre-render when possible.
        if_statement_opening_body = self.get_expression_control_node_if_statement_opening_body(model)
        if_statement_success_closing_body = self.get_expression_control_node_if_statement_success_closing_body(model)
        if_statement_failure_closing_body = self.get_expression_control_node_if_statement_failure_closing_body(model)

        # Render the expression control node if statement template.
        return self.expression_control_node_if_statement_template.render(
            in_line_statement=in_line_statement,
            if_statement_opening_body=if_statement_opening_body,
            if_statement_success_closing_body=if_statement_success_closing_body,
            if_statement_failure_closing_body=if_statement_failure_closing_body,
            expression_control_node_success_closing_body=expression_control_node_success_closing_body,
            nested_statement=nested_statement
        )

    def render_expression_control_node_body_expression(
            self,
            model: Expression,
            enforce_no_method_creation: bool,
            expression_control_node_success_closing_body: str
    ) -> str:
        """Render the desired expression body of the control node."""
        return self.render_expression_control_node_body_default(
            model, enforce_no_method_creation, expression_control_node_success_closing_body
        )

    def render_expression_control_node_body_default(
            self,
            model: Union[Primary, VariableRef, Expression],
            enforce_no_method_creation: bool,
            expression_control_node_success_closing_body: str
    ) -> str:
        """Render the desired primary/variable reference body of the control node."""
        return self.render_expression_control_node_if_statement(
            model, enforce_no_method_creation, expression_control_node_success_closing_body
        )

    def render_expression_control_node_body(
            self, model: Union[Expression, Primary, VariableRef], enforce_no_method_creation: bool
    ) -> str:
        """Render the desired body of the control node."""
        # Pre-render data when possible.
        expression_control_node_opening_body = self.get_expression_control_node_opening_body(model)
        expression_control_node_success_closing_body = self.get_expression_control_node_success_closing_body(model)
        expression_control_node_failure_closing_body = self.get_expression_control_node_failure_closing_body(model)

        # Control the manner in which the body is rendered.
        contains_closing_body = \
            expression_control_node_success_closing_body != "" or expression_control_node_failure_closing_body != ""
        equivalent_to_true = model.is_true() and expression_control_node_failure_closing_body == ""
        equivalent_to_false = model.is_false() and expression_control_node_success_closing_body == ""

        # Render the conditional body.
        if not contains_closing_body or equivalent_to_true or equivalent_to_false:
            # The conditional body can be in-line.
            conditional_body = self.get_expression_control_node_in_line_statement(model, enforce_no_method_creation)
        elif isinstance(model, Expression):
            # Use the expression control node body renderer (needed due to VerCors short-circuit evaluation issues).
            conditional_body = self.render_expression_control_node_body_expression(
                model, enforce_no_method_creation, expression_control_node_success_closing_body
            )
        else:
            # Use the default control node body renderer.
            conditional_body = self.render_expression_control_node_body_default(
                model, enforce_no_method_creation, expression_control_node_success_closing_body
            )

        # Render the control node body.
        return self.expression_control_node_body_template.render(
            conditional_body=conditional_body,
            expression_control_node_opening_body=expression_control_node_opening_body,
            expression_control_node_success_closing_body=expression_control_node_success_closing_body,
            expression_control_node_failure_closing_body=expression_control_node_failure_closing_body,
            contains_closing_body=contains_closing_body,
            equivalent_to_true=equivalent_to_true,
            equivalent_to_false=equivalent_to_false
        )

    def render_expression_control_node(
            self, model: Union[Expression, Primary, VariableRef], enforce_no_method_creation: bool = False
    ) -> str:
        """Render the given statement object as in-line Java code and create supportive methods if necessary."""
        # Determine if the control node can be in-line or not--if not, generate a new function and refer to it.
        if not enforce_no_method_creation and self.requires_atomic_node_method(model):
            # Render the body of the expression control node.
            control_node_body = self.render_expression_control_node_body(model, enforce_no_method_creation)

            # Give the node a name, render it as a separate method, and return a call to the method.
            control_node_name = self.get_control_node_name()

            # Pre-render applicable information and data.
            human_readable_expression_identification = self.get_expression_control_node_comment(model)

            # Render the following sections at the last moment to ensure that all recursive steps have finished.
            expression_control_node_contract = self.get_expression_control_node_contract(model)

            # Render the statement as a control node method using the control node template.
            self.current_control_node_methods.append(
                self.expression_control_node_template.render(
                    human_readable_expression_identification=human_readable_expression_identification,
                    control_node_name=control_node_name,
                    control_node_body=control_node_body,
                    expression_control_node_contract=expression_control_node_contract
                )
            )

            # Return a call to the generated method.
            return f"{control_node_name}()"
        else:
            # Get the in-line statement and return it outright.
            in_line_statement = self.get_expression_control_node_in_line_statement(model, enforce_no_method_creation)

            # Return the statement as an in-line Java statement.
            return in_line_statement


    # def render_expression_control_node_old(self, model: SlcoStatementNode, enforce_no_method_creation: bool = False) -> str:
    #     """Render the given statement object as in-line Java code and create supportive methods if necessary."""
    #     # Construct the in-line statement.
    #     if isinstance(model, Expression):
    #         in_line_statement = self.render_expression(model, enforce_no_method_creation)
    #     elif isinstance(model, Primary):
    #         in_line_statement = self.render_primary(model, enforce_no_method_creation)
    #     elif isinstance(model, VariableRef):
    #         in_line_statement = self.render_variable_ref(model)
    #     else:
    #         raise Exception(f"No function exists to turn objects of type {type(model)} into in-line Java statements.")
    #
    #     # Determine if the control node can be in-line or not--if not, generate a new function and refer to it.
    #     if not enforce_no_method_creation and self.requires_atomic_node_method(model):
    #         # Give the node a name, render it as a separate method, and return a call to the method.
    #         control_node_name = self.get_control_node_name()
    #
    #         # Determine if the internal if-statement can be simplified.
    #         condition_is_true = model.is_true()
    #         condition_is_false = not condition_is_true and model.is_false()
    #
    #         # Pre-render applicable information and data.
    #         human_readable_expression_identification = self.get_expression_control_node_comment(model)
    #         has_lock_operations_in_exit_nodes = \
    #             model.locking_atomic_node.success_exit.has_locks() or model.locking_atomic_node.failure_exit.has_locks()
    #         has_single_exit_path = condition_is_true or condition_is_false
    #
    #         # Render the following sections at the last moment to ensure that all recursive steps have finished.
    #         expression_control_node_contract = self.get_expression_control_node_contract(model)
    #         expression_control_node_opening_body = self.get_expression_control_node_opening_body(model)
    #         expression_control_node_success_closing_body = self.get_expression_control_node_success_closing_body(model)
    #         expression_control_node_failure_closing_body = self.get_expression_control_node_failure_closing_body(model)
    #
    #         # Render the statement as a control node method using the control node template.
    #         self.current_control_node_methods.append(
    #             self.expression_control_node_template.render(
    #                 control_node_name=control_node_name,
    #                 in_line_statement=in_line_statement,
    #                 condition_is_true=condition_is_true,
    #                 condition_is_false=condition_is_false,
    #                 human_readable_expression_identification=human_readable_expression_identification,
    #                 expression_control_node_contract=expression_control_node_contract,
    #                 expression_control_node_opening_body=expression_control_node_opening_body,
    #                 expression_control_node_success_closing_body=expression_control_node_success_closing_body,
    #                 expression_control_node_failure_closing_body=expression_control_node_failure_closing_body,
    #                 has_lock_operations_in_exit_nodes=has_lock_operations_in_exit_nodes,
    #                 has_single_exit_path=has_single_exit_path
    #             )
    #         )
    #
    #         # Have the in-line statement call the control node method instead.
    #         in_line_statement = f"{control_node_name}()"
    #
    #     # Return the statement as an in-line Java statement.
    #     return in_line_statement

    def get_statement_prefix(self) -> str:
        """Get an unique prefix for the statement."""
        i = self.current_control_node_id
        self.current_control_node_id += 1
        return f"{self.current_transition_prefix}_s_{i}"

    @staticmethod
    def get_root_expression_comment(is_superfluous: bool, model: Expression) -> str:
        """
        Create an easily identifiable comment for root expression statements.
        """
        # Create an easily identifiable comment.
        original_slco_expression_string = str(model.get_original_statement())
        preprocessed_slco_expression_string = str(model)
        statement_comment = "// "
        if is_superfluous:
            statement_comment += "(Superfluous) "
        statement_comment += f"SLCO expression | {original_slco_expression_string}"
        if original_slco_expression_string != preprocessed_slco_expression_string:
            statement_comment += f" -> {preprocessed_slco_expression_string}"
        return statement_comment

    # noinspection PyMethodMayBeStatic
    def get_root_expression_opening_body(self, model: Union[Expression, Primary]) -> str:
        """Get the opening statements of the root expression object."""
        return ""

    # noinspection PyMethodMayBeStatic
    def get_root_expression_closing_body(self, model: Union[Expression, Primary]) -> str:
        """Get the closing statements of the root expression object."""
        return ""

    def render_root_expression(self, model: Union[Expression, Primary]) -> str:
        """Render the given expression as an if statement in Java code."""
        # Create an unique prefix for the statement.
        self.current_statement_prefix = self.get_statement_prefix()

        # Support flags to control rendering settings.
        is_superfluous = model.is_true() and not model.locking_atomic_node.has_locks()

        # Create an in-line Java code string for the expression.
        in_line_expression = ""
        if not is_superfluous:
            in_line_expression = self.render_expression_control_node(model)

        # Pre-render applicable information and data.
        human_readable_expression_identification = self.get_root_expression_comment(is_superfluous, model)

        # Render the following sections at the last moment to ensure that all recursive steps have finished.
        root_expression_opening_body = self.get_root_expression_opening_body(model)
        root_expression_closing_body = self.get_root_expression_closing_body(model)

        # Render the assignment template.
        return self.expression_template.render(
            in_line_expression=in_line_expression,
            human_readable_expression_identification=human_readable_expression_identification,
            root_expression_opening_body=root_expression_opening_body,
            root_expression_closing_body=root_expression_closing_body,
            is_superfluous=is_superfluous
        )

    @staticmethod
    def get_assignment_comment(model: Assignment) -> str:
        """Create an easily identifiable comment for assignment statements."""
        # Create an easily identifiable comment.
        original_slco_statement_string = str(model.get_original_statement())
        preprocessed_slco_statement_string = str(model)
        statement_comment = f"// SLCO assignment | {original_slco_statement_string}"
        if original_slco_statement_string != preprocessed_slco_statement_string:
            statement_comment += f" -> {preprocessed_slco_statement_string}"
        return statement_comment

    # noinspection PyMethodMayBeStatic
    def get_assignment_opening_body(self, model: Assignment) -> str:
        """Get the opening statements of the assignment object."""
        entry_locking_instruction = self.render_locking_instruction(model.locking_atomic_node.entry_node)
        locking_check = self.render_locking_check(model)
        return "\n".join(v for v in [entry_locking_instruction, locking_check] if v != "")

    # noinspection PyMethodMayBeStatic
    def get_assignment_closing_body(self, model: Assignment) -> str:
        """Get the closing statements of the assignment object."""
        success_exit_locking_instruction = self.render_locking_instruction(model.locking_atomic_node.success_exit)
        return success_exit_locking_instruction

    def render_assignment(self, model: Assignment) -> str:
        """Render the given assignment as Java code."""
        # TODO: properly handle byte calculations. Maybe use char instead of int with a & 0xff mask?
        #   - Do expressions need to be changed too to handle this functionality?

        # Create an unique prefix for the statement.
        self.current_statement_prefix = self.get_statement_prefix()

        # Create an in-line Java code string for the left and right hand side.
        in_line_lhs = self.render_expression_control_node(model.left)
        in_line_rhs = self.render_expression_control_node(model.right)

        # Pre-render applicable information and data.
        human_readable_assignment_identification = self.get_assignment_comment(model)

        # Render the following sections at the last moment to ensure that all recursive steps have finished.
        assignment_opening_body = self.get_assignment_opening_body(model)
        assignment_closing_body = self.get_assignment_closing_body(model)

        # Render the assignment template.
        return self.assignment_template.render(
            in_line_lhs=in_line_lhs,
            in_line_rhs=in_line_rhs,
            human_readable_assignment_identification=human_readable_assignment_identification,
            assignment_opening_body=assignment_opening_body,
            assignment_closing_body=assignment_closing_body,
            is_byte_typed=model.left.var.is_byte
        )

    @staticmethod
    def get_composite_comment(model: Composite) -> str:
        """Create an easily identifiable comment for composite statements."""
        # Create an easily identifiable comment.
        original_slco_composite_string = str(model.get_original_statement())
        preprocessed_slco_composite_string = str(model)
        statement_comment = f"// SLCO composite | {original_slco_composite_string}"
        if original_slco_composite_string != preprocessed_slco_composite_string:
            statement_comment += f" -> {preprocessed_slco_composite_string}"
        return statement_comment

    # noinspection PyMethodMayBeStatic
    def get_composite_opening_body(self, model: Composite) -> str:
        """Get the opening statements of the composite object."""
        return ""

    # noinspection PyMethodMayBeStatic
    def get_composite_closing_body(self, model: Composite) -> str:
        """Get the closing statements of the composite object."""
        return ""

    def render_composite(self, model: Composite) -> str:
        """Render the given composite object as Java code."""
        # Pre-render all statements used in the composite.
        rendered_statements = [self.render_root_expression(model.guard)]
        for a in model.assignments:
            rendered_statements.append(self.render_assignment(a))

        # Pre-render applicable information and data.
        human_readable_composite_identification = self.get_composite_comment(model)

        # Render the following sections at the last moment to ensure that all recursive steps have finished.
        composite_opening_body = self.get_composite_opening_body(model)
        composite_closing_body = self.get_composite_closing_body(model)

        # Render the composite template.
        return self.composite_template.render(
            human_readable_composite_identification=human_readable_composite_identification,
            rendered_statements=rendered_statements,
            composite_opening_body=composite_opening_body,
            composite_closing_body=composite_closing_body
        )

    # noinspection PyMethodMayBeStatic
    def get_transition_contract(self, model: Transition) -> str:
        """Get the contract of the transition method."""
        return ""

    # noinspection PyMethodMayBeStatic
    def get_transition_opening_body(self, model: Transition) -> str:
        """Get the opening statements of the transition method."""
        return ""

    # noinspection PyMethodMayBeStatic
    def get_transition_closing_body(self, model: Transition) -> str:
        """Get the closing statements of the transition method."""
        return f"currentState = { self.current_class.name }_{ self.current_state_machine.name }Thread.States.{ model.target };"

    # noinspection PyMethodMayBeStatic
    def render_excluded_transition(self, model: Transition) -> str:
        """Render the given excluded SLCO transition as Java code."""
        return f"// SLCO transition {model} " \
               f"| Excluded from decision structure due to a false guard or unreachable code."

    def render_included_transition(self, model: Transition) -> str:
        """Render the given included SLCO transition as Java code."""
        # Reset the current supporting variables.
        self.current_transition_prefix = f"t_{model.source}_{model.id}"
        self.current_control_node_id = 0
        self.current_control_node_methods = []

        # Render each of the statements in sequence.
        rendered_statements = []
        for s in model.statements:
            # Note that the first statement in the transition will always be a guard statement.
            if isinstance(s, Composite):
                result = self.render_composite(s)
            elif isinstance(s, Assignment):
                result = self.render_assignment(s)
            elif isinstance(s, (Expression, Primary)):
                result = self.render_root_expression(s)
            else:
                raise Exception(f"No function exists to turn objects of type {type(s)} into Java statements.")
            if result is not None:
                rendered_statements.append(result)

        # Pre-render applicable information and data.
        human_readable_transition_identification = str(model)

        # Render the following sections at the last moment to ensure that all recursive steps have finished.
        transition_contract = self.get_transition_contract(model)
        transition_opening_body = self.get_transition_opening_body(model)
        transition_closing_body = self.get_transition_closing_body(model)

        # Render the transition template.
        return self.transition_template.render(
            control_node_methods=self.current_control_node_methods,
            human_readable_transition_identification=human_readable_transition_identification,
            model_source=model.source,
            model_target=model.target,
            model_id=model.id,
            rendered_statements=rendered_statements,
            state_machine_name=self.current_state_machine.name,
            class_name=self.current_class.name,
            transition_contract=transition_contract,
            transition_opening_body=transition_opening_body,
            transition_closing_body=transition_closing_body
        )

    def render_transition(self, model: Transition) -> str:
        """Render the SLCO transition as Java code."""
        # Set the current transition.
        self.current_transition = model

        if model.is_excluded:
            return self.render_excluded_transition(model)
        else:
            return self.render_included_transition(model)

    def render_transition_call(self, model: Transition) -> str:
        """Render a call to the given transition as Java code."""
        human_readable_transition_identification = str(model)

        # Render a call to the specified transition.
        return self.transition_call_template.render(
            human_readable_transition_identification=human_readable_transition_identification,
            model_source=model.source,
            model_id=model.id,
        )

    def render_deterministic_decision_node(self, model: DecisionNode) -> str:
        """Render the given deterministic decision node as Java code."""
        rendered_decisions, rendered_excluded_transitions = self.get_rendered_nested_decision_structures(model)
        return self.deterministic_decision_node_template.render(
            rendered_decisions=rendered_decisions,
            rendered_excluded_transitions=rendered_excluded_transitions
        )

    def render_sequential_decision_node(self, model: DecisionNode) -> str:
        """Render the given sequential decision node as Java code."""
        rendered_decisions, rendered_excluded_transitions = self.get_rendered_nested_decision_structures(model)
        return self.sequential_decision_node_template.render(
            rendered_decisions=rendered_decisions,
            rendered_excluded_transitions=rendered_excluded_transitions
        )

    def render_pick_random_decision_node(self, model: DecisionNode) -> str:
        """Render the given pick random decision node as Java code."""
        rendered_decisions, rendered_excluded_transitions = self.get_rendered_nested_decision_structures(model)
        return self.pick_random_decision_node_template.render(
            rendered_decisions=rendered_decisions,
            rendered_excluded_transitions=rendered_excluded_transitions
        )

    def render_non_deterministic_decision_node(self, model: DecisionNode) -> str:
        """Render the given non-deterministic decision node as Java code."""
        if settings.non_determinism:
            return self.render_non_deterministic_decision_node(model)
        else:
            return self.render_sequential_decision_node(model)

    def render_decision_node(self, model: DecisionNode) -> str:
        """Render the given decision node as Java code."""
        if model.is_deterministic:
            return self.render_deterministic_decision_node(model)
        else:
            return self.render_non_deterministic_decision_node(model)

    def get_rendered_nested_decision_structures(self, model: DecisionNode) -> Tuple[List[str], List[str]]:
        """Render the nested decision structures of the given model and report them as lists."""
        # Pre-render nested decision structures.
        rendered_decisions = []
        for decision in model.decisions:
            if isinstance(decision, Transition):
                result = self.render_transition_call(decision)
            elif isinstance(decision, DecisionNode):
                result = self.render_decision_node(decision)
            else:
                raise Exception(
                    f"No function exists to turn objects of type {type(decision)} into in-line Java statements."
                )
            rendered_decisions.append(result)
        # Include comments for transitions that have been purposefully excluded by the decision node.
        rendered_excluded_transitions = []
        for t in model.excluded_transitions:
            rendered_excluded_transitions.append(f"// - {t}")

        # Return the two lists.
        return rendered_decisions, rendered_excluded_transitions

    # noinspection PyMethodMayBeStatic
    def get_decision_structure_contract(self, model: StateMachine, state: State) -> str:
        """Get the contract of the decision structure method."""
        return ""

    # noinspection PyMethodMayBeStatic
    def get_decision_structure_opening_body(self, model: StateMachine, state: State) -> str:
        """Get the opening statements of the decision structure method."""
        return ""

    # noinspection PyMethodMayBeStatic
    def get_decision_structure_closing_body(self, model: StateMachine, state: State) -> str:
        """Get the closing statements of the decision structure method."""
        return ""

    def render_decision_structure(self, model: StateMachine, state: State) -> str:
        """Render the decision structure for the given state machine and starting state as Java code."""
        # Find the target decision structure and pre-render it as Java code.
        if state not in model.state_to_decision_node:
            method_body = f"// There are no transitions starting in state {state}."
        else:
            root_target_node: DecisionNode = model.state_to_decision_node[state]
            method_body = self.render_decision_node(root_target_node)

        # Pre-render supporting statements.
        # Render the following sections at the last moment to ensure that all recursive steps have finished.
        decision_structure_contract = self.get_decision_structure_contract(model, state)
        decision_structure_opening_body = self.get_decision_structure_opening_body(model, state)
        decision_structure_closing_body = self.get_decision_structure_closing_body(model, state)

        # Render the state's decision structure as a method.
        return self.decision_structure_template.render(
            state=state,
            method_body=method_body,
            decision_structure_contract=decision_structure_contract,
            decision_structure_opening_body=decision_structure_opening_body,
            decision_structure_closing_body=decision_structure_closing_body
        )

    # noinspection PyMethodMayBeStatic
    def render_variable_type(self, model: Variable, include_modifiers=False) -> str:
        """Render the type of the given variable object."""
        type_name = "boolean" if model.is_boolean else "char" if model.is_byte else "int"
        if model.is_array:
            return f"final {type_name}[]" if include_modifiers else f"{type_name}[]"
        else:
            return f"volatile {type_name}" if include_modifiers else f"{type_name}"

    def render_variable_default_value(self, model: Variable) -> str:
        """Render the default value of the given variable."""
        a = model.def_values if model.is_array else model.def_value
        if model.is_array:
            default_value = f"new {self.render_variable_type(model)} {{ {', '.join(map(str, a))} }}"
        elif model.is_byte:
            default_value = f"(char) {a}"
        else:
            default_value = a
        return default_value

    def render_state_machine_constructor_contract(self, model: StateMachine) -> str:
        """Get the contract of the state machine's constructor."""
        return ""

    def render_state_machine(self, model: StateMachine) -> str:
        """Render the SLCO state machine as Java code."""
        # Set the current state machine.
        self.current_state_machine = model

        # Pre-render the state machine components.
        states = [str(s) for s in model.states]
        variable_declarations = [f"{self.render_variable_type(v)} {v.name}" for v in model.variables]
        initial_state = str(model.initial_state)
        lock_ids_array_size = model.lock_ids_list_size
        target_locks_array_size = model.target_locks_list_size
        variable_instantiations = [f"{v.name} = {self.render_variable_default_value(v)}" for v in model.variables]
        transitions = [self.render_transition(t) for t in model.transitions]
        decision_structures = [self.render_decision_structure(model, s) for s in model.states]

        # Render the following sections at the last moment to ensure that all recursive steps have finished.
        constructor_contract = self.render_state_machine_constructor_contract(model)

        # Render the state machine template.
        return self.state_machine_template.render(
            model_name=model.name,
            class_name=model.parent.name,
            states=states,
            initial_state=initial_state,
            variable_declarations=variable_declarations,
            lock_ids_array_size=lock_ids_array_size,
            target_locks_array_size=target_locks_array_size,
            variable_instantiations=variable_instantiations,
            transitions=transitions,
            decision_structures=decision_structures,
            settings=settings,
            constructor_contract=constructor_contract
        )

    def render_class_constructor_contract(self, model: Class) -> str:
        """Get the contract of the class's constructor."""
        return ""

    def render_class(self, model: Class) -> str:
        """Render the SLCO class as Java code."""
        # Set the current class.
        self.current_class = model

        # Pre-render the class components.
        state_machine_names = [sm.name for sm in model.state_machines]
        state_machines = [self.render_state_machine(sm) for sm in model.state_machines]
        variable_declarations = [f"{self.render_variable_type(v, True)} {v.name}" for v in model.variables]
        constructor_arguments = [f"{self.render_variable_type(v)} {v.name}" for v in model.variables]
        variable_names = [f"{v.name}" for v in model.variables]
        lock_array_size = max((v.lock_id + v.type.size for v in model.variables), default=0)

        # Render the following sections at the last moment to ensure that all recursive steps have finished.
        constructor_contract = self.render_class_constructor_contract(model)

        # Render the class template.
        return self.class_template.render(
            model_name=model.name,
            state_machine_names=state_machine_names,
            state_machines=state_machines,
            variable_declarations=variable_declarations,
            constructor_arguments=constructor_arguments,
            variable_names=variable_names,
            lock_array_size=lock_array_size,
            constructor_contract=constructor_contract
        )

    def render_object_instantiation(self, model: Object) -> str:
        """Render the instantiation of the given object."""
        arguments = []
        for i, a in enumerate(model.initial_values):
            v = model.type.variables[i]
            if v.is_array:
                arguments.append(f"new {self.render_variable_type(v)}{{ {', '.join(map(str, a)).lower()} }}")
            elif v.is_byte:
                arguments.append(f"(char) {a}")
            else:
                arguments.append(a)

        # Render the object instantiation template.
        return self.object_instantiation.render(
            name=model.type.name,
            arguments=arguments
        )

    def render_lock_manager(self) -> str:
        """Render the lock manager of the model."""
        # Render the lock manager template.
        return self.lock_manager_template.render(
            settings=settings
        )

    def render_model_constructor_contract(self, model: SlcoModel) -> str:
        """Get the contract of the model's constructor."""
        return ""

    def render_model(self, model: SlcoModel) -> str:
        """Render the SLCO model as Java code."""
        # Pre-render the contained classes, lock manager and object instantiations.
        lock_manager = self.render_lock_manager()
        classes = [self.render_class(c) for c in model.classes]
        object_instantiations = [self.render_object_instantiation(o) for o in model.objects]

        # Render the following sections at the last moment to ensure that all recursive steps have finished.
        constructor_contract = self.render_model_constructor_contract(model)

        # Render the model template.
        return self.model_template.render(
            model_name=model.name,
            lock_manager=lock_manager,
            classes=classes,
            object_instantiations=object_instantiations,
            constructor_contract=constructor_contract
        )
