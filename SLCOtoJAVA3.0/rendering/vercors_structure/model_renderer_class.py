from __future__ import annotations

from typing import List, Dict, Tuple

from objects.ast.interfaces import SlcoStatementNode
from objects.ast.models import SlcoModel, Class, StateMachine, Transition, Variable, VariableRef
from rendering.java.model_renderer import JavaModelRenderer


class VercorsModelRenderer(JavaModelRenderer):
    """
    Create a subclass of the java model renderer that renders VerCors verification statements.
    """

    def __init__(self):
        super().__init__()

        # Create additional supportive variables.
        self.verification_targets: Dict[Variable, List[int]] = dict()

        # Overwrite the model, class and state machine templates to completely stripped down versions without nesting.
        self.state_machine_template = self.env.get_template("vercors/state_machine.jinja2template")
        self.class_template = self.env.get_template("vercors/class.jinja2template")
        self.model_template = self.env.get_template("vercors/model.jinja2template")

        # Add additional templates.
        self.shared_permissions_template = self.env.get_template("vercors/util/shared_permissions.jinja2template")
        self.vercors_contract_template = self.env.get_template("vercors/util/vercors_contract.jinja2template")
        self.state_machine_constructor_contract_template = self.env.get_template(
            "vercors/state_machine_constructor_contract.jinja2template"
        )
        self.class_constructor_contract_template = self.env.get_template(
            "vercors/class_constructor_contract.jinja2template"
        )

    def render_variable_ref(self, model: VariableRef) -> str:
        result = super().render_variable_ref(model)
        if model.var.is_class_variable:
            # Prepend the variable name with a c, since the class variables needs to be referenced through an object.
            return f"c.{result}"
        return result

    # TODO: render and/or differently.

    def render_vercors_contract_permissions(self) -> str:
        """Render the permissions that need to be included in every transition and control method contract."""
        # Pre-render data.
        state_machine_array_variable_names = [v.name for v in self.current_state_machine.variables if v.is_array]
        state_machine_array_variable_lengths = [v.type.size for v in self.current_state_machine.variables if v.is_array]
        state_machine_array_variable_permissions = list(
            zip(state_machine_array_variable_names, state_machine_array_variable_lengths)
        )
        state_machine_variable_permissions = [
            f"{v.name}[*]" if v.is_array else v.name for v in self.current_state_machine.variables
        ]

        # Render the shared permissions template.
        return self.shared_permissions_template.render(
            state_machine_array_variable_permissions=state_machine_array_variable_permissions,
            state_machine_variable_permissions=state_machine_variable_permissions
        )

    def render_vercors_expression_control_node_contract_body(self, model: SlcoStatementNode) -> Tuple[str, List[str]]:
        """Render the body of the vercors contract of the given transition, including potential pure functions."""
        # Pre-render data.
        vercors_contract_permissions = self.render_vercors_contract_permissions()
        return vercors_contract_permissions, []

    def render_vercors_expression_control_node_contract(self, model: SlcoStatementNode) -> str:
        """Render the vercors contract of the given transition."""
        # Pre-render data.
        contract_body, pure_functions = self.render_vercors_expression_control_node_contract_body(model)

        # Render the vercors contract template.
        return self.vercors_contract_template.render(
            pure_functions=pure_functions,
            contract_body=contract_body
        )

    def get_expression_control_node_contract(self, model: SlcoStatementNode) -> str:
        return self.render_vercors_expression_control_node_contract(model)

    @staticmethod
    def requires_atomic_node_method(model: SlcoStatementNode) -> bool:
        # Render all atomic node holding expressions as a function--this is required due to a need to verify
        # conjunctions and disjunctions in a specific way because of limitations in VerCors.
        return model.locking_atomic_node is not None

    def render_vercors_transition_contract_body(self, model: Transition) -> Tuple[str, List[str]]:
        """Render the body of the vercors contract of the given transition, including potential pure functions."""
        # Pre-render data.
        vercors_contract_permissions = self.render_vercors_contract_permissions()
        return vercors_contract_permissions, []

    def render_vercors_transition_contract(self, model: Transition) -> str:
        """Render the vercors contract of the given transition."""
        # Pre-render data.
        contract_body, pure_functions = self.render_vercors_transition_contract_body(model)

        # Render the vercors contract template.
        return self.vercors_contract_template.render(
            pure_functions=pure_functions,
            contract_body=contract_body
        )

    def get_transition_contract(self, model: Transition) -> str:
        return self.render_vercors_transition_contract(model)

    def get_transition_closing_body(self, model: Transition) -> str:
        # Do not render the state change, since enums are not supported by VerCors.
        return ""

    def render_transition(self, model: Transition) -> str:
        # Keep a dictionary of assignment operations that need to be verified by the transition's contract.
        # This variable is used later in the transition's contract.
        self.verification_targets: Dict[Variable, List[int]] = dict()
        return super().render_transition(model)

    def render_state_machine_constructor_contract(self, model: StateMachine) -> str:
        return self.state_machine_constructor_contract_template.render()

    def render_class_constructor_contract(self, model: Class) -> str:
        # Pre-render data.
        variable_names = [f"{v.name}" for v in model.variables]
        array_variable_names = [f"{v.name}" for v in model.variables if v.is_array]

        # Render the class constructor template.
        return self.class_constructor_contract_template.render(
            variable_names=variable_names,
            array_variable_names=array_variable_names
        )

    def render_model_constructor_contract(self, model: SlcoModel) -> str:
        # TODO: render the appropriate permission statements and validations.
        return super().render_model_constructor_contract(model)



