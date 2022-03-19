from __future__ import annotations

from typing import List, Union

import settings
from objects.ast.interfaces import SlcoStatementNode
from objects.ast.models import SlcoModel, Transition, Expression, Primary, StateMachine, State, Assignment
from objects.locking.models import LockingNode
from rendering.java.renderer import JavaModelRenderer


class LogMeasurementsModelRenderer(JavaModelRenderer):
    """
    Create a subclass of the Java model renderer that renders log statements for gathering performance metrics.
    """

    def __init__(self):
        super().__init__()

        # Add additional templates.
        self.logger_variable_and_static_initialization_template = self.env.get_template(
            "measurements/logger_variable_and_static_initialization.jinja2template"
        )
        self.sleep_template = self.env.get_template(
            "measurements/sleep.jinja2template"
        )

    # LOG STATEMENTS.
    def get_transition_identifier(self, model: Transition):
        """Get a string with which transitions can be identified in the logs."""
        return f"{self.current_class.name} {self.current_state_machine.name} {model.id} {model.source} {model.target}"

    # def render_locking_instruction(self, model: LockingNode) -> str:
    #     result = super().render_locking_instruction(model)
    #     if model.has_locks() and not settings.statement_level_locking:
    #         locking_instructions = model.locking_instructions
    #         result = self.join_with_strip([
    #             f"logger.info(\"T.L {self.get_transition_identifier(self.current_transition)} "
    #             f"{len(locking_instructions.locks_to_acquire)} {len(locking_instructions.unpacked_lock_requests)} "
    #             f"{len(locking_instructions.locks_to_release)}\");",
    #             result,
    #         ])
    #     return result
    #
    # def get_expression_control_node_opening_body(self, model: SlcoStatementNode) -> str:
    #     result = super().get_expression_control_node_opening_body(model)
    #     return self.join_with_strip([
    #             f"logger.info(\"T.CN {self.get_transition_identifier(self.current_transition)} "
    #             f"{self.current_control_node_id - 1} {len(self.current_control_node_methods)}\");",
    #             result,
    #         ])
    #
    # def get_assignment_opening_body(self, model: Assignment) -> str:
    #     result = super().get_assignment_opening_body(model)
    #     return self.join_with_strip([
    #             f"logger.info(\"T.A {self.get_transition_identifier(self.current_transition)} "
    #             f"{self.current_control_node_id - 1}\");",
    #             result,
    #         ])

    def get_transition_call_opening_body(self, model: Transition) -> str:
        result = super().get_transition_call_opening_body(model)
        return self.join_with_strip([
                result,
                f"logger.info(\"T.O {self.get_transition_identifier(model)}\");"
            ])

    def get_transition_call_success_closing_body(self, model: Transition) -> str:
        result = super().get_transition_call_success_closing_body(model)
        return self.join_with_strip([
                result,
                f"logger.info(\"T.CS {self.get_transition_identifier(model)}\");"
            ])

    def get_transition_call_failure_closing_body(self, model: Transition) -> str:
        result = super().get_transition_call_failure_closing_body(model)
        return self.join_with_strip([
                result,
                f"logger.info(\"T.CF {self.get_transition_identifier(model)}\");"
            ])

    def get_decision_structure_opening_body(self, model: StateMachine, state: State) -> str:
        result = super().get_decision_structure_opening_body(model, state)
        return self.join_with_strip([
                result,
                f"logger.info(\"D.O {self.current_class.name} {self.current_state_machine.name} {state}\");"
            ])

    def get_decision_structure_closing_body(self, model: StateMachine, state: State) -> str:
        result = super().get_decision_structure_closing_body(model, state)
        return self.join_with_strip([
                result,
                f"logger.info(\"D.CF {self.current_class.name} {self.current_state_machine.name} {state}\");"
            ])

    # LOGGER INITIALIZATION.
    def get_model_support_variables(self, model: SlcoModel) -> List[str]:
        # Add the logger initializer to the model.
        result = super().get_model_support_variables(model)
        result.append(
            self.logger_variable_and_static_initialization_template.render(
                model_name=model.name,
                log_settings=settings.settings_abbreviations
            )
        )
        return result

    def get_import_statements(self) -> List[str]:
        # Add the imports needed for the logging system.
        result = super().get_import_statements()
        result.extend([
            "import org.apache.logging.log4j.LogManager;",
            "import org.apache.logging.log4j.Logger;",
            "import org.apache.logging.log4j.core.lookup.MainMapLookup;",
            "import java.time.format.DateTimeFormatter;",
            "import java.time.Instant;",
        ])
        return result

    def get_main_supportive_closing_method_calls(self) -> List[str]:
        result = super().get_main_supportive_closing_method_calls()
        return result + [self.sleep_template.render()]
