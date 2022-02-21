from __future__ import annotations

from typing import List, Tuple

from objects.ast.interfaces import SlcoStatementNode
from objects.ast.models import SlcoModel, Class, StateMachine, Transition, VariableRef, Expression, State
from rendering.java.model_renderer import JavaModelRenderer


class VercorsModelRenderer(JavaModelRenderer):
    """
    Create a subclass of the java model renderer that renders VerCors verification statements.
    """

    def __init__(self):
        super().__init__()

        # Create additional supportive variables.
        self.current_assumptions: List[str] = []

        # Overwrite the model, class and state machine templates to completely stripped down versions without nesting.
        self.state_machine_constructor_body_template = self.env.get_template(
            "vercors/state_machine_constructor_body.jinja2template"
        )
        self.state_machine_variable_declarations_template = self.env.get_template(
            "vercors/state_machine_variable_declarations.jinja2template"
        )
        self.state_machine_template = self.env.get_template("vercors/state_machine.jinja2template")
        self.class_template = self.env.get_template("vercors/class.jinja2template")
        self.model_template = self.env.get_template("vercors/model.jinja2template")

        # Add additional templates.
        self.vercors_contract_template = self.env.get_template("vercors/util/vercors_contract.jinja2template")
        self.state_machine_permissions_template = self.env.get_template(
            "vercors/util/state_machine_permissions.jinja2template"
        )
        self.class_permissions_template = self.env.get_template(
            "vercors/util/class_permissions.jinja2template"
        )
        self.common_permissions_template = self.env.get_template(
            "vercors/util/common_permissions.jinja2template"
        )
        self.vercors_assumptions_template = self.env.get_template(
            "vercors/util/vercors_assumptions.jinja2template"
        )
        self.vercors_expression_control_node_contract_body_template = self.env.get_template(
            "vercors/util/vercors_expression_control_node_contract_body.jinja2template"
        )
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

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_if_statement_opening_body(self, model: SlcoStatementNode) -> str:
        # TODO
        return super().get_expression_control_node_if_statement_opening_body(model)

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_if_statement_success_closing_body(self, model: SlcoStatementNode) -> str:
        # TODO
        # Get the in-line statement.
        in_line_statement = self.get_expression_control_node_in_line_statement(model, enforce_no_method_creation=True)
        return f"//@ assert {in_line_statement};"

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_if_statement_failure_closing_body(self, model: SlcoStatementNode) -> str:
        # TODO
        # Get the in-line statement.
        in_line_statement = self.get_expression_control_node_in_line_statement(model, enforce_no_method_creation=True)
        return f"//@ assert !({in_line_statement});"

    def render_expression_control_node_body_expression_conjunction(
            self, model: Expression, enforce_no_method_creation: bool, expression_control_node_success_closing_body: str
    ) -> str:
        """Render a conjunction statement as nested if-statements to circumvent the lack of short-circuit evaluation."""
        # Render a control node for each of the values.
        value_in_line_statements = []
        for v in model.values:
            value_in_line_statements.append((v, self.render_expression_control_node(v, enforce_no_method_creation)))

            # Add v as an assumption, since it is guaranteed to hold from this point on due to atomicity.
            in_line_statement = self.get_expression_control_node_in_line_statement(
                v, enforce_no_method_creation=True
            )
            self.current_assumptions.append(in_line_statement)

        # Remove all assumptions associated to this conjunction, since they can only be used locally.
        self.current_assumptions = self.current_assumptions[:-len(model.values)]

        # Render the if-statements with the given in-line statement and combine them through a nested if-structure.
        conditional_body = self.render_expression_control_node_if_statement(
            value_in_line_statements[-1][0],
            enforce_no_method_creation,
            expression_control_node_success_closing_body,
            in_line_statement=value_in_line_statements[-1][1]
        ).strip()
        for v, in_line_statement in reversed(value_in_line_statements[:-1]):
            conditional_body = self.render_expression_control_node_if_statement(
                v, enforce_no_method_creation, nested_statement=conditional_body, in_line_statement=in_line_statement
            ).strip()
        return conditional_body

    def render_expression_control_node_body_expression_disjunction(
            self, model: Expression, enforce_no_method_creation: bool, expression_control_node_success_closing_body: str
    ) -> str:
        """Render a disjunction statement as if-statements to circumvent the lack of short-circuit evaluation."""
        # Render a control node for each of the values.
        value_in_line_statements = []
        for v in model.values:
            value_in_line_statements.append((v, self.render_expression_control_node(v, enforce_no_method_creation)))

            # Add !(v) as an assumption, since it is guaranteed not to hold from this point on due to atomicity.
            in_line_statement = self.get_expression_control_node_in_line_statement(
                v, enforce_no_method_creation=True
            )
            self.current_assumptions.append(f"!({in_line_statement})")

        # Remove all assumptions associated to this disjunction, since they can only be used locally.
        self.current_assumptions = self.current_assumptions[:-len(model.values)]

        # Render the if-statements with the given in-line statement and combine them in succession.
        value_if_statements = [
            self.render_expression_control_node_if_statement(
                v,
                enforce_no_method_creation,
                expression_control_node_success_closing_body,
                in_line_statement=in_line_statement
            ).strip() for v, in_line_statement in value_in_line_statements
        ]
        conditional_body = "\n".join(value_if_statements)
        return conditional_body

    def render_expression_control_node_body_expression(
        self, model: Expression, enforce_no_method_creation: bool, expression_control_node_success_closing_body: str
    ) -> str:
        if model.op == "and":
            return self.render_expression_control_node_body_expression_conjunction(
                model, enforce_no_method_creation, expression_control_node_success_closing_body
            )
        elif model.op == "or":
            return self.render_expression_control_node_body_expression_disjunction(
                model, enforce_no_method_creation, expression_control_node_success_closing_body
            )
        else:
            # Render all other statements in the default way.
            return super().render_expression_control_node_body_expression(
                model, enforce_no_method_creation, expression_control_node_success_closing_body
            )

    def render_vercors_contract_class_permissions(self) -> str:
        """
        Render the class variable permissions that need to be included in all statement containing contracts.
        """
        # Pre-render data.
        class_array_variable_names = [v.name for v in self.current_class.variables if v.is_array]
        class_array_variable_lengths = [v.type.size for v in self.current_class.variables if v.is_array]
        class_array_variable_permissions = list(
            zip(class_array_variable_names, class_array_variable_lengths)
        )
        class_variable_permissions = [
            f"c.{v.name}[*]" if v.is_array else f"c.{v.name}" for v in self.current_class.variables
        ]

        # Render the class permissions template.
        return self.class_permissions_template.render(
            class_array_variable_permissions=class_array_variable_permissions,
            class_variable_permissions=class_variable_permissions
        )

    def render_vercors_contract_state_machine_permissions(self) -> str:
        """
        Render the state machine variable permissions that need to be included in all statement containing contracts.
        """
        # Pre-render data.
        state_machine_array_variable_names = [v.name for v in self.current_state_machine.variables if v.is_array]
        state_machine_array_variable_lengths = [v.type.size for v in self.current_state_machine.variables if v.is_array]
        state_machine_array_variable_permissions = list(
            zip(state_machine_array_variable_names, state_machine_array_variable_lengths)
        )
        state_machine_variable_permissions = [
            f"{v.name}[*]" if v.is_array else v.name for v in self.current_state_machine.variables
        ]

        # Render the state machine permissions template.
        return self.state_machine_permissions_template.render(
            state_machine_array_variable_permissions=state_machine_array_variable_permissions,
            state_machine_variable_permissions=state_machine_variable_permissions
        )

    def render_vercors_contract_common_permissions(self) -> str:
        """
        Render the base variable permissions that need to be included in all statement containing contracts.
        """
        # Render the common permissions template.
        return self.common_permissions_template.render()

    def render_vercors_expression_control_node_contract_requirements(
            self, model: SlcoStatementNode
    ) -> Tuple[str, List[str]]:
        """
        Render additional requirements that should be included in the contract.
        """
        # Gather the appropriate data.
        vercors_assumptions = self.current_assumptions

        # Render the vercors assumptions template.
        return self.vercors_assumptions_template.render(
            vercors_assumptions=vercors_assumptions
        ), []

    def render_vercors_expression_control_node_contract_body(self, model: SlcoStatementNode) -> Tuple[str, List[str]]:
        """Render the body of the vercors contract of the given transition, including potential pure functions."""
        # Pre-render data.
        in_line_statement = self.get_expression_control_node_in_line_statement(model, enforce_no_method_creation=True)
        vercors_common_contract_permissions = self.render_vercors_contract_common_permissions()
        vercors_state_machine_contract_permissions = self.render_vercors_contract_state_machine_permissions()
        vercors_class_contract_permissions = self.render_vercors_contract_class_permissions()
        vercors_requirements, pure_functions = self.render_vercors_expression_control_node_contract_requirements(model)

        # Render the vercors expression control node contract body template.
        return self.vercors_expression_control_node_contract_body_template.render(
            in_line_statement=in_line_statement,
            vercors_common_contract_permissions=vercors_common_contract_permissions,
            vercors_state_machine_contract_permissions=vercors_state_machine_contract_permissions,
            vercors_class_contract_permissions=vercors_class_contract_permissions,
            vercors_requirements=vercors_requirements
        ), pure_functions

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

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_success_closing_body(self, model: SlcoStatementNode) -> str:
        # Render the in-line statement as an assertion but only if a conjunction or disjunction.
        result = super().get_expression_control_node_success_closing_body(model)
        if isinstance(model, Expression) and model.op in ["and", "or"]:
            in_line_statement = self.get_expression_control_node_in_line_statement(
                model, enforce_no_method_creation=True
            )
            return "\n".join(v for v in [f"//@ assert {in_line_statement};", result] if v != "")
        else:
            return result

    # noinspection PyMethodMayBeStatic
    def get_expression_control_node_failure_closing_body(self, model: SlcoStatementNode) -> str:
        result = super().get_expression_control_node_failure_closing_body(model)
        if isinstance(model, Expression) and model.op in ["and", "or"]:
            in_line_statement = self.get_expression_control_node_in_line_statement(
                model, enforce_no_method_creation=True
            )
            return "\n".join(v for v in [f"//@ assert !({in_line_statement});", result] if v != "")
        else:
            return result

    def render_vercors_transition_contract_body(self, model: Transition) -> Tuple[str, List[str]]:
        """Render the body of the vercors contract of the given transition, including potential pure functions."""
        # Pre-render data.
        vercors_common_contract_permissions = self.render_vercors_contract_common_permissions().strip()
        vercors_state_machine_contract_permissions = self.render_vercors_contract_state_machine_permissions().strip()
        vercors_class_contract_permissions = self.render_vercors_contract_class_permissions().strip()
        vercors_contract_permissions = "\n\n".join(v for v in [
            vercors_common_contract_permissions,
            vercors_state_machine_contract_permissions,
            vercors_class_contract_permissions
        ] if v != "")

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

    def get_decision_structure_contract_body(self, model: StateMachine, state: State) -> Tuple[str, List[str]]:
        """
        Render the body of the vercors contract of the given decision structure, including potential pure functions.
        """
        # Pre-render data.
        vercors_common_contract_permissions = self.render_vercors_contract_common_permissions().strip()
        vercors_state_machine_contract_permissions = self.render_vercors_contract_state_machine_permissions().strip()
        vercors_class_contract_permissions = self.render_vercors_contract_class_permissions().strip()
        vercors_contract_permissions = "\n\n".join(v for v in [
            vercors_common_contract_permissions,
            vercors_state_machine_contract_permissions,
            vercors_class_contract_permissions
        ] if v != "")

        return vercors_contract_permissions, []

    def get_decision_structure_contract(self, model: StateMachine, state: State) -> str:
        # Pre-render data.
        contract_body, pure_functions = self.get_decision_structure_contract_body(model, state)

        # Render the vercors contract template.
        return self.vercors_contract_template.render(
            pure_functions=pure_functions,
            contract_body=contract_body
        )

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



