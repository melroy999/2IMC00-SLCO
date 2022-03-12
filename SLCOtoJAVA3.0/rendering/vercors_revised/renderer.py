from __future__ import annotations

from typing import List, Tuple, Set

import networkx as nx

from objects.ast.interfaces import SlcoStatementNode
from objects.ast.models import SlcoModel, Class, StateMachine, Transition, VariableRef, Expression, State, Assignment, \
    Composite, Variable
from objects.ast.util import get_variable_references
from objects.locking.models import LockingNode
from rendering.java.renderer import JavaModelRenderer


class VercorsModelRendererRevised(JavaModelRenderer):
    """
    Create a subclass of the java model renderer that facilitates the rendering of VerCors verification statements.
    """
    def __init__(self):
        super().__init__()

        # Overwrite the model, class and state machine templates to completely stripped down versions without nesting.
        self.state_machine_constructor_body_template = self.env.get_template(
            "vercors_revised/state_machine_constructor_body.jinja2template"
        )
        self.state_machine_variable_declarations_template = self.env.get_template(
            "vercors_revised/state_machine_variable_declarations.jinja2template"
        )
        self.state_machine_template = self.env.get_template("vercors_revised/state_machine.jinja2template")
        self.class_template = self.env.get_template("vercors_revised/class.jinja2template")
        self.model_template = self.env.get_template("vercors_revised/model.jinja2template")

        # Add additional templates.
        self.common_permissions_template = self.env.get_template(
            "vercors_revised/util/common_permissions.jinja2template"
        )
        self.base_class_permissions_template = self.env.get_template(
            "vercors_revised/util/base_class_permissions.jinja2template"
        )
        self.base_state_machine_permissions_template = self.env.get_template(
            "vercors_revised/util/base_state_machine_permissions.jinja2template"
        )
        self.method_contract_template = self.env.get_template(
            "vercors_revised/util/method_contract.jinja2template"
        )
        self.state_machine_constructor_contract_template = self.env.get_template(
            "vercors_revised/state_machine_constructor_contract.jinja2template"
        )
        self.class_constructor_contract_template = self.env.get_template(
            "vercors_revised/class_constructor_contract.jinja2template"
        )
        self.range_check_assumption_method_template = self.env.get_template(
            "vercors_revised/range_check_assumption.jinja2template"
        )

    # TODO Temporary overwrites.
    def render_locking_instruction(self, model: LockingNode) -> str:
        return ""

    def render_locking_check(self, model: SlcoStatementNode) -> str:
        return ""



    def get_plain_text_in_line_statement(self, model: SlcoStatementNode) -> str:
        """Render the given statement as an in-line statement with no control nodes."""
        return self.get_expression_control_node_in_line_statement(model, enforce_no_method_creation=True)

    def render_variable_ref(self, model: VariableRef) -> str:
        # Prepend the variable name with a c, since the class variables needs to be referenced through an object.
        result = super().render_variable_ref(model)
        return f"c.{result}" if model.var.is_class_variable else result

    @staticmethod
    def requires_atomic_node_method(model: SlcoStatementNode) -> bool:
        # Render all atomic node holding expressions as a function--this is required due to a need to verify
        # conjunctions and disjunctions in a specific way because of limitations in VerCors.
        return model.locking_atomic_node is not None

    # RANGE CHECK METHODS.
    #     - Adds methods that generate the data required to render the appropriate range checks for array variables.
    #         - Provides a method that orders variable references topologically to circumvent order related errors.
    #         - Provides a method that renders a range check statement for a particular element.
    @staticmethod
    def get_topologically_ordered_variable_references(
            model: SlcoStatementNode, variable_references: Set[VariableRef] = None
    ) -> List[VariableRef]:
        """
        Order the given variable references and the variables used within their indices in topological order with
        respect to the variables used within indices.
        """
        # Gather all references variables if no variable references list is given.
        if variable_references is None:
            variable_references = get_variable_references(model)

        # return variable_references
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
        return ordered_references

    def get_range_check_variable_references(self, model: SlcoStatementNode) -> List[VariableRef]:
        """Get the variables contained within the given variable reference that require range checking."""
        return [v for v in self.get_topologically_ordered_variable_references(model) if v.var.is_array]

    def render_range_check_statement(self, model: VariableRef, operator: str = "&&") -> str:
        """Render the range check statement associated with the given variable."""
        in_line_statement = self.get_plain_text_in_line_statement(model.index)
        return f"0 <= {in_line_statement} {operator} {in_line_statement} < {model.var.type.size}"

    def render_range_check_body_statements(self, model: SlcoStatementNode) -> str:
        """Render the appropriate range check assumptions for the given statement to be included in a body contract."""
        # Find all variables that need to be range checked.
        range_check_targets = self.get_range_check_variable_references(model)

        # Return an empty string if no range checks need to be performed.
        if len(range_check_targets) == 0:
            return ""

        # Generate the range check statements and combine them.
        range_check_comment = "// Assume that all of the accessed indices are within range."
        range_check_statements = [f"//@ assume {self.render_range_check_statement(s)};" for s in range_check_targets]
        return "\n".join([range_check_comment] + range_check_statements)

    def insert_range_check_assumption_method(self, model: SlcoStatementNode, transition_call: bool = False) -> str:
        """Render the appropriate range check assumptions for the given statement within a method to make them local."""
        range_check_assumption_body = self.render_range_check_body_statements(model)
        range_check_assumption_contract = self.render_range_check_assumption_method_contract(model)
        if transition_call:
            # The range check assumption is rendered by the transition, and hence the name needs to be easy to trace.
            range_check_assumption_name = f"range_check_assumption_t_{self.current_transition.id}"
        else:
            # TODO: create a proper name.
            range_check_assumption_name = ""

        # Render the range check assumption method template.
        self.current_control_node_methods.append(
            self.range_check_assumption_method_template.render(
                range_check_assumption_body=range_check_assumption_body,
                range_check_assumption_contract=range_check_assumption_contract,
                range_check_assumption_name=range_check_assumption_name
            )
        )

        # Return the method name.
        return f"{range_check_assumption_name}();"

    # WORKAROUNDS FOR SHORT-CIRCUIT EVALUATIONS.
    #     - Overwrite the control node body expression rendering function to include a fix for short-circuit evaluation.
    #     - Render conjunctions as a nested if-structure.
    #     - Render disjunctions as a collection of if-structures.
    def render_expression_control_node_body_expression_conjunction(
            self, model: Expression, enforce_no_method_creation: bool, expression_control_node_success_closing_body: str
    ) -> str:
        """Render a conjunction statement as nested if-statements to circumvent the lack of short-circuit evaluation."""
        # Render a control node for each of the values.
        value_in_line_statements = []
        for v in model.values:
            value_in_line_statements.append((v, self.render_expression_control_node(v, enforce_no_method_creation)))

        # Select the last statement.
        last_statement = value_in_line_statements[-1]
        last_statement_target = last_statement[0]
        last_statement_in_line_statement = last_statement[1]

        # Render the last statement in the conjunction as an if-statement.
        conditional_body = self.render_expression_control_node_if_statement(
            last_statement_target,
            enforce_no_method_creation,
            expression_control_node_success_closing_body,
            in_line_statement=last_statement_in_line_statement
        ).strip()

        # Render the conjunction as a nested if-structure.
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

        # Render the disjunction as a collection of if-statements.
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

    # CONTRACT RENDERING METHODS.
    # TODO: comments
    def render_common_contract_permissions(self) -> str:
        """Render the base variable permissions that need to be included in all contracts."""
        # Render the common permissions template.
        return self.common_permissions_template.render()

    @staticmethod
    def render_contract_permissions(variables: List[Variable], renderer) -> str:
        """Render the permissions and assertions that are needed to access the array variables."""
        # Gather the target variables.
        target_variables = [v for v in variables if v.is_array]

        # Return an empty string if no target variables are present.
        if len(target_variables) == 0:
            return ""

        # Use the template to render the permissions.
        variable_names = [v.name for v in target_variables]
        variable_lengths = [v.type.size for v in target_variables]
        target_entries = list(zip(variable_names, variable_lengths))
        return renderer.render(
            target_entries=target_entries
        )

    def render_base_state_machine_contract_permissions(self) -> str:
        """
        Render the permissions and assertions that are needed to access the current state machine's array variables.
        """
        return self.render_contract_permissions(
            self.current_state_machine.variables,
            self.base_state_machine_permissions_template
        )

    def render_full_state_machine_contract_permissions(self) -> str:
        """Render permissions granting full access to all variables contained within the current state machine."""
        # Gather the target variables.
        target_variables = [f"{v.name}[*]" if v.is_array else v.name for v in self.current_state_machine.variables]

        # Return an empty string if no target variables are present.
        if len(target_variables) == 0:
            return ""

        # Generate the permission statements and combine them.
        permission_comment = "// Require and ensure the permission of writing to all state machine variables."
        permission_statements = [f"context Perm({v}, 1);" for v in target_variables]
        return "\n".join([permission_comment] + permission_statements)

    def render_base_class_contract_permissions(self) -> str:
        """
        Render the permissions and assertions that are needed to access the current class's array variables.
        """
        return self.render_contract_permissions(
            self.current_class.variables,
            self.base_class_permissions_template
        )

    def render_full_class_contract_permissions(self) -> str:
        """Render permissions granting full access to all variables contained within the current class."""
        # Gather the target variables.
        target_variables = [f"{v.name}[*]" if v.is_array else f"{v.name}" for v in self.current_class.variables]

        # Return an empty string if no target variables are present.
        if len(target_variables) == 0:
            return ""

        # Generate the permission statements and combine them.
        permission_comment = "// Require and ensure the permission of writing to all class variables."
        permission_statements = [f"context Perm(c.{v}, 1);" for v in target_variables]
        return "\n".join([permission_comment] + permission_statements)

    def render_range_check_contract_statements(self, model: SlcoStatementNode, scope: str = "context") -> str:
        """Render the appropriate range checks for the given statement to be included in a method contract."""
        # Find all variables that need to be range checked.
        range_check_targets = self.get_range_check_variable_references(model)

        # Return an empty string if no range checks need to be performed.
        if len(range_check_targets) == 0:
            return ""

        # Generate the range check statements and combine them.
        range_check_comment = "// Require and ensure that all of the accessed indices are within range."
        range_check_statements = [f"{scope} {self.render_range_check_statement(s)};" for s in range_check_targets]
        return "\n".join([range_check_comment] + range_check_statements)

    def render_result_contract_statement(self, model: SlcoStatementNode, use_old_values: bool = False):
        """Render code that checks if the result of the function is equivalent to the target statement."""
        result_check_comment = "// Ensure that the result of the function is equivalent to the target statement."
        function_call = "\\old" if use_old_values else ""
        result_check_statement = f"ensures \\result == {function_call}({self.get_plain_text_in_line_statement(model)});"
        return "\n".join([result_check_comment, result_check_statement])

    @staticmethod
    def render_values_unaltered_statement(model: Variable) -> str:
        """Render a statement for the given variable that ensures that the old value is equivalent to the new."""
        v_name = model.name
        if model.is_class_variable:
            v_name = f"c.{v_name}"
        if model.is_array:
            return f"(\\forall* int _i; 0 <= _i && _i < {v_name}.length; {v_name}[_i] == \\old({v_name}[_i]))"
        else:
            return f"{v_name} == \\old({v_name})"

    def render_state_machine_contract_values_unaltered_statements(self) -> str:
        """Render code that ensures that none of the class variable values are changed by the method."""
        target_variables = self.current_state_machine.variables

        # Return an empty string if no variables are present.
        if len(target_variables) == 0:
            return ""

        range_check_comment = "// Ensure that all state machine variable values remain unchanged."
        range_check_statements = [f"ensures {self.render_values_unaltered_statement(v)};" for v in target_variables]
        return "\n".join([range_check_comment] + range_check_statements)

    def render_class_contract_values_unaltered_statements(self) -> str:
        """Render code that ensures that none of the class variable values are changed by the method."""
        target_variables = self.current_class.variables

        # Return an empty string if no variables are present.
        if len(target_variables) == 0:
            return ""

        range_check_comment = "// Ensure that all class variable values remain unchanged."
        range_check_statements = [f"ensures {self.render_values_unaltered_statement(v)};" for v in target_variables]
        return "\n".join([range_check_comment] + range_check_statements)

    # VERIFICATION INSERTS.
    # TODO: comments
    def get_expression_control_node_success_closing_body(self, model: SlcoStatementNode) -> str:
        result = super().get_expression_control_node_success_closing_body(model)
        if isinstance(model, Expression) and model.op in ["or", "and"]:
            result = "\n".join(v.strip() for v in [result, "// Short-circuit fix trigger."] if v.strip() != "")
        return result

    def get_transition_closing_body(self, model: Transition) -> str:
        # Do not render the state change, since enums are not supported by VerCors.
        return ""

    def get_transition_call_opening_body(self, model: Transition) -> str:
        # Add the appropriate assumptions needed to call the function.
        result = super().get_transition_call_opening_body(model)
        range_check_assumption_method_name = f"//@ ghost range_check_assumption_t_{self.current_transition.id}();"
        return "\n".join(v.strip() for v in [result, range_check_assumption_method_name] if v.strip() != "")

    # CONTRACT RENDERING METHODS.
    # TODO: comments
    def render_method_contract(self, target_statements: List[str], pure_functions: List[str]) -> str:
        """Render a method contract containing the given statements and pure functions."""
        contract_body = "\n\n".join(v.strip() for v in target_statements if v.strip() != "")

        # Render the method contract template.
        return self.method_contract_template.render(
            contract_body=contract_body,
            pure_functions=pure_functions,
        )

    # noinspection PyUnusedLocal
    def get_range_check_assumption_contract_entries(self, model: SlcoStatementNode) -> Tuple[List[str], List[str]]:
        """Get the contract statements that are needed for the range check assumption methods."""
        return [
            self.render_common_contract_permissions(),
            self.render_base_state_machine_contract_permissions(),
            self.render_full_state_machine_contract_permissions(),
            self.render_base_class_contract_permissions(),
            self.render_full_class_contract_permissions(),
            self.render_range_check_contract_statements(model, scope="ensures"),
            self.render_state_machine_contract_values_unaltered_statements(),
            self.render_class_contract_values_unaltered_statements()
        ], []

    def render_range_check_assumption_method_contract(self, model: SlcoStatementNode) -> str:
        """Render the vercors contract of the given rance check assumption method."""
        target_statements, pure_functions = self.get_range_check_assumption_contract_entries(model)
        return self.render_method_contract(target_statements, pure_functions)

    def get_expression_control_node_contract_entries(self, model: SlcoStatementNode) -> Tuple[List[str], List[str]]:
        """Get the contract statements that are needed for control node methods."""
        return [
            self.render_common_contract_permissions(),
            self.render_base_state_machine_contract_permissions(),
            self.render_full_state_machine_contract_permissions(),
            self.render_base_class_contract_permissions(),
            self.render_full_class_contract_permissions(),
            self.render_range_check_contract_statements(model, scope="context"),
            self.render_result_contract_statement(model),
            self.render_state_machine_contract_values_unaltered_statements(),
            self.render_class_contract_values_unaltered_statements()
        ], []

    def render_expression_control_node_contract(self, model: SlcoStatementNode) -> str:
        """Render the vercors contract of the given transition."""
        target_statements, pure_functions = self.get_expression_control_node_contract_entries(model)
        return self.render_method_contract(target_statements, pure_functions)

    def get_expression_control_node_contract(self, model: SlcoStatementNode) -> str:
        return self.render_expression_control_node_contract(model)

    def get_transition_contract_entries(self, model: Transition) -> Tuple[List[str], List[str]]:
        """Get the contract statements that are needed for transition methods."""
        guard_expression = model.guard.guard if isinstance(model.guard, Composite) else model.guard
        # In addition, render a range assumption method.
        self.insert_range_check_assumption_method(guard_expression, transition_call=True)
        return [
            self.render_common_contract_permissions(),
            self.render_base_state_machine_contract_permissions(),
            self.render_full_state_machine_contract_permissions(),
            self.render_base_class_contract_permissions(),
            self.render_full_class_contract_permissions(),
            self.render_range_check_contract_statements(guard_expression, scope="requires"),
            self.render_result_contract_statement(guard_expression, use_old_values=True)
        ], []

    def render_transition_contract(self, model: Transition) -> str:
        """Render the vercors contract of the given transition."""
        target_statements, pure_functions = self.get_transition_contract_entries(model)
        return self.render_method_contract(target_statements, pure_functions)

    def get_transition_contract(self, model: Transition) -> str:
        return self.render_transition_contract(model)

    # noinspection PyUnusedLocal
    def get_decision_structure_contract_entries(self, model: StateMachine, state: State) -> Tuple[List[str], List[str]]:
        """Get the contract statements that are needed for decision structure methods."""
        return [
            self.render_common_contract_permissions(),
            self.render_base_state_machine_contract_permissions(),
            self.render_full_state_machine_contract_permissions(),
            self.render_base_class_contract_permissions(),
            self.render_full_class_contract_permissions()
        ], []

    def render_decision_structure_contract(self, model: StateMachine, state: State) -> str:
        """Render the vercors contract of the given decision structure."""
        target_statements, pure_functions = self.get_decision_structure_contract_entries(model, state)
        return self.render_method_contract(target_statements, pure_functions)

    def get_decision_structure_contract(self, model: StateMachine, state: State) -> str:
        # Pre-render data.
        return self.render_decision_structure_contract(model, state)

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
        return super().render_model_constructor_contract(model)
