from __future__ import annotations

from typing import List

import settings
from objects.ast.models import SlcoModel, Transition, StateMachine, State
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
        self.force_rollover_template = self.env.get_template(
            "measurements/force_rollover.jinja2template"
        )

    # LOG STATEMENTS.
    def get_decision_structure_identifier(self, model: State):
        """Get a string with which decision structures can be identified in the logs."""
        return f"{self.current_class.name}.{self.current_state_machine.name}.{model}"

    def get_transition_identifier(self, model: Transition):
        """Get a string with which transitions can be identified in the logs."""
        return f"{self.get_decision_structure_identifier(model.source)}.{model.target}.{model.id}"

    def get_transition_call_opening_body(self, model: Transition) -> str:
        result = super().get_transition_call_opening_body(model)
        return self.join_with_strip([
                result,
                f"logger.info(\"{self.get_transition_identifier(model)}.O\");"
            ])

    def get_transition_call_success_closing_body(self, model: Transition) -> str:
        result = super().get_transition_call_success_closing_body(model)
        return self.join_with_strip([
                result,
                f"logger.info(\"{self.get_transition_identifier(model)}.CS\");",
                f"logger.info(\"{self.get_decision_structure_identifier(model.source)}.CS\");"
            ])

    def get_transition_call_failure_closing_body(self, model: Transition) -> str:
        result = super().get_transition_call_failure_closing_body(model)
        return self.join_with_strip([
                result,
                f"logger.info(\"{self.get_transition_identifier(model)}.CF\");"
            ])

    def get_decision_structure_opening_body(self, model: StateMachine, state: State) -> str:
        result = super().get_decision_structure_opening_body(model, state)
        return self.join_with_strip([
                result,
                f"logger.info(\"{self.get_decision_structure_identifier(state)}.O\");"
            ])

    def get_decision_structure_closing_body(self, model: StateMachine, state: State) -> str:
        result = super().get_decision_structure_closing_body(model, state)
        return self.join_with_strip([
                result,
                f"logger.info(\"{self.get_decision_structure_identifier(state)}.CF\");"
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
            "import org.apache.logging.log4j.core.Appender;",
            "import org.apache.logging.log4j.core.LoggerContext;",
            "import org.apache.logging.log4j.core.appender.RollingRandomAccessFileAppender;",
            "import org.apache.logging.log4j.core.lookup.MainMapLookup;",
            "import java.time.format.DateTimeFormatter;",
            "import java.time.Instant;",
        ])
        return result

    def get_main_supportive_closing_method_calls(self) -> List[str]:
        result = super().get_main_supportive_closing_method_calls()
        return result + [
            self.sleep_template.render(),
            self.force_rollover_template.render(),
            self.sleep_template.render(),
        ]
