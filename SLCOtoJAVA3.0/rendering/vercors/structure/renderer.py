from typing import Tuple, List, Union, Dict

from objects.ast.interfaces import SlcoStatementNode
from objects.ast.models import Variable, Expression, Primary, Assignment, Transition
from objects.locking.models import LockingNode
from rendering.vercors.renderer import VercorsModelRenderer


class VercorsStructureModelRenderer(VercorsModelRenderer):
    """
    Create a subclass of the java model renderer that renders VerCors verification statements to verify the correctness
    of the rendered models from a structural point of view.
    """

    def __init__(self):
        super().__init__()

        # Create additional supportive variables.
        self.verification_targets: Dict[Variable, List[int]] = dict()
        self.assignment_number: int = 0

        # Add additional templates.
        self.change_restrictions_template = self.env.get_template(
            "vercors/structure/value_change_restrictions.jinja2template"
        )
        self.value_check_pure_function_template = self.env.get_template(
            "vercors/structure/value_check_pure_function.jinja2template"
        )
        self.transition_value_change_check_template = self.env.get_template(
            "vercors/structure/transition_value_change_check.jinja2template"
        )

    def render_locking_instruction(self, model: LockingNode) -> str:
        # The structure check does not consider locking--exclude the code from the render.
        return ""

    def render_locking_check(self, model: SlcoStatementNode) -> str:
        # The structure check does not consider locking--exclude the code from the render.
        return ""

    @staticmethod
    def render_vercors_no_values_changed_statement(model: Variable) -> str:
        """Render a statement for the given variable that ensures that the old value is equivalent to the new."""
        variable_name = model.name
        if model.is_class_variable:
            variable_name = f"c.{variable_name}"
        if model.is_array:
            return f"(\\forall* int _i; 0 <= _i && _i < {variable_name}.length; " \
                   f"{variable_name}[_i] == \\old({variable_name}[_i]))"
        else:
            return f"{variable_name} == \\old({variable_name})"

    def render_vercors_expression_control_node_no_values_changed_requirements(
            self, model: SlcoStatementNode
    ) -> str:
        """Render vercors statements that verify that all values remain unchanged."""
        # Pre-render data.
        class_value_change_restrictions = [
            self.render_vercors_no_values_changed_statement(v) for v in self.current_class.variables
        ]
        state_machine_value_change_restrictions = [
            self.render_vercors_no_values_changed_statement(v) for v in self.current_state_machine.variables
        ]

        # Render the vercors change restrictions template.
        return self.change_restrictions_template.render(
            class_value_change_restrictions=class_value_change_restrictions,
            state_machine_value_change_restrictions=state_machine_value_change_restrictions
        )

    def render_vercors_expression_control_node_contract_requirements(
            self, model: SlcoStatementNode
    ) -> Tuple[str, List[str]]:
        # Add variable change restrictions to the contract as defined by the SLCO semantics.
        requirements, pure_functions = super().render_vercors_expression_control_node_contract_requirements(model)
        change_restrictions = self.render_vercors_expression_control_node_no_values_changed_requirements(model)
        return "\n\n".join(v.strip() for v in [change_restrictions, requirements] if v != ""), pure_functions

    def get_root_expression_opening_body(self, model: Union[Expression, Primary]) -> str:
        result = super().get_root_expression_opening_body(model)

        # Save the evaluation of the guard statement for evaluation purposes.
        in_line_statement = self.get_expression_control_node_in_line_statement(
            model, enforce_no_method_creation=True
        )
        return "\n".join(v for v in [f"//@ ghost _guard = ({in_line_statement});", result] if v != "")

    def get_root_expression_success_closing_body(self, model: Union[Expression, Primary]) -> str:
        result = super().get_root_expression_success_closing_body(model)

        # Check whether the original statement holds in the success branch.
        in_line_statement = self.get_expression_control_node_in_line_statement(
            model.get_original_statement(), enforce_no_method_creation=True
        )
        return "\n".join(v for v in [f"//@ assert {in_line_statement};", result] if v != "")

    def get_root_expression_failure_closing_body(self, model: Union[Expression, Primary]) -> str:
        result = super().get_root_expression_failure_closing_body(model)

        # Check whether the original statement holds in the failure branch.
        in_line_statement = self.get_expression_control_node_in_line_statement(
            model.get_original_statement(), enforce_no_method_creation=True
        )
        return "\n".join(v for v in [f"//@ assert !({in_line_statement});", result] if v != "")

    def get_assignment_opening_body(self, model: Assignment) -> str:
        result = super().get_assignment_opening_body(model)

        # Statements that should be added in the opening body.
        statements = []

        # Save the value assigned to the variable for verification purposes.
        in_line_rhs = self.get_expression_control_node_in_line_statement(model.right, enforce_no_method_creation=True)
        statements.append(f"//@ ghost _rhs_{self.assignment_number} = {in_line_rhs};")

        # Check whether the right side of the assignment is equivalent to the original statement if the renders differ.
        in_line_rhs_original = self.get_expression_control_node_in_line_statement(
            model.right.get_original_statement(), enforce_no_method_creation=True
        )
        if in_line_rhs != in_line_rhs_original:
            statements.append(f"//@ assert ({in_line_rhs}) == ({in_line_rhs_original});")

        if model.left.var.is_array:
            # Save the value used as an index for the target variable verification purposes.
            in_line_index = self.get_expression_control_node_in_line_statement(
                model.left.index, enforce_no_method_creation=True
            )
            statements.append(f"//@ ghost _index_{self.assignment_number} = {in_line_index};")

            # Check whether the used index is equivalent to the original index if the renders differ.
            in_line_index_original = self.get_expression_control_node_in_line_statement(
                model.left.index.get_original_statement(), enforce_no_method_creation=True
            )
            if in_line_index != in_line_index_original:
                statements.append(f"//@ assert ({in_line_index}) == ({in_line_index_original});")
        return "\n".join(v for v in [result] + statements if v != "")

    def get_assignment_closing_body(self, model: Assignment) -> str:
        result = super().get_assignment_closing_body(model)

        # Check whether the appropriate value has been assigned.
        rhs_target = f"_rhs_{self.assignment_number}"
        lhs_target = model.left.var.name
        if model.left.var.is_class_variable:
            lhs_target = f"c.{lhs_target}"
        if model.left.var.is_array:
            lhs_target = f"{lhs_target}[_index_{self.assignment_number}]"
        return "\n".join(v for v in [f"//@ assert {lhs_target} == {rhs_target};", result] if v != "")

    def render_assignment(self, model: Assignment) -> str:
        try:
            return super().render_assignment(model)
        finally:
            # Add the assignment as a verification target.
            self.verification_targets[model.left.var].append(self.assignment_number)

            # Increment the current assignment number after rendering an assignment.
            self.assignment_number += 1

    def render_vercors_pure_function(self, pure_function_prefix: str, v: Variable, targets: List[int]):
        """
        Render a pure function that checks for the expected value changes for the given variable and assignment ids.
        """
        pure_function_type = "boolean" if v.is_boolean else "int"
        pure_function_name = f"{pure_function_prefix}_{v.name}"
        pure_function_target_parameters = ", ".join(
            f"int _index_{i}, {pure_function_type} _rhs_{i}" for i in targets
        )
        # Note that the last assignment will be the first evaluated in the nested if structure.
        pure_function_body = f"(_i == _index_{targets[0]}) ? _rhs_{targets[0]} : v_old"
        for i in targets[1:]:
            pure_function_body = f"(_i == _index_{i}) ? _rhs_{i} : ({pure_function_body})"

        # Render the value check pure function template.
        return self.value_check_pure_function_template.render(
            pure_function_type=pure_function_type,
            pure_function_name=pure_function_name,
            pure_function_target_parameters=pure_function_target_parameters,
            pure_function_body=pure_function_body,
        )

    @staticmethod
    def render_vercors_pure_function_call(pure_function_prefix: str, v: Variable, targets: List[int]):
        """Render a call to the appropriate pure function."""
        pure_function_name = f"{pure_function_prefix}_{v.name}"
        pure_function_parameters = ", ".join(f"_index_{i}, _rhs_{i}" for i in targets)
        variable_name = f"c.{v.name}" if v.is_class_variable else v.name
        return f"{pure_function_name}(_i, {pure_function_parameters}, \\old({variable_name}[_i]))"

    def render_vercors_transition_value_check_contract(self, model: Transition) -> Tuple[str, List[str]]:
        """Render the VerCors statements that verify whether the appropriate values are changed by the transition."""
        # Gather a list of support variables used by the statements within the transition.
        support_variables: List[str] = ["boolean _guard"]
        for v, targets in self.verification_targets.items():
            type_name = "boolean" if v.is_boolean else "int"
            for i in targets:
                support_variables.append(f"{type_name} _rhs_{i}")
                if v.is_array:
                    support_variables.append(f"{type_name} _index_{i}")

        # Create rules for the expected outcome of the transition, based on the guard statement.
        return_value_verification = "\\result == _guard"

        # Create pure functions to verify the correctness of value changes of array variables with.
        value_check_pure_functions = []
        pure_function_prefix = f"value_{model.source}_{model.id}"
        for v, targets in self.verification_targets.items():
            if v.is_array and len(targets) > 0:
                value_check_pure_functions.append(self.render_vercors_pure_function(pure_function_prefix, v, targets))

        # Create rules for the expected changes in variable values.
        value_change_verification_rules = []
        for v, targets in self.verification_targets.items():
            variable_name = f"c.{v.name}" if v.is_class_variable else v.name
            if v.is_array:
                if len(targets) == 0:
                    value_change_verification_rules.append(
                        f"(\\forall* int _i; 0 <= _i && _i < {variable_name}.length; "
                        f"{variable_name}[_i] == \\old({variable_name}[_i]))"
                    )
                else:
                    pure_function_call = self.render_vercors_pure_function_call(pure_function_prefix, v, targets)
                    value_change_verification_rules.append(
                        f"_guard ==> (\\forall* int _i; 0 <= _i && _i < {variable_name}.length; "
                        f"{variable_name}[_i] == {pure_function_call})"
                    )
                    value_change_verification_rules.append(
                        f"!_guard ==> (\\forall* int _i; 0 <= _i && _i < {variable_name}.length; "
                        f"{variable_name}[_i] == \\old({variable_name}[_i]))"
                    )
            else:
                if len(targets) == 0:
                    value_change_verification_rules.append(f"{variable_name} == \\old({variable_name})")
                else:
                    value_change_verification_rules.append(f"_guard ==> ({variable_name} == _rhs_{targets[-1]})")
                    value_change_verification_rules.append(f"!_guard ==> ({variable_name} == \\old({variable_name}))")

        # Render the value check pure function template.
        return self.transition_value_change_check_template.render(
            support_variables=support_variables,
            return_value_verification=return_value_verification,
            value_change_verification_rules=value_change_verification_rules
        ), value_check_pure_functions

    def render_vercors_transition_contract_body(self, model: Transition) -> Tuple[str, List[str]]:
        # Include statements that verify whether the transition changes the values appropriately.
        result, result_pure_functions = super().render_vercors_transition_contract_body(model)
        transition_value_check_contract, pure_functions = self.render_vercors_transition_value_check_contract(model)
        pure_functions = pure_functions + result_pure_functions
        return "\n\n".join(v for v in [result, transition_value_check_contract] if v != ""), pure_functions

    def render_transition(self, model: Transition) -> str:
        # Reset the dictionary of assignment operations that need to be verified by the transition's contract.
        self.verification_targets = {
            v: [] for v in self.current_class.variables + self.current_state_machine.variables
        }

        # Reset the current assignment number when starting a transition.
        self.assignment_number = 0
        return super().render_transition(model)

