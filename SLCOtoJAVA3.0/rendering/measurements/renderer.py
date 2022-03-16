from __future__ import annotations

from typing import List, Union

from objects.ast.models import SlcoModel, Transition, Expression, Primary, StateMachine, State
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

    # LOG STATEMENTS.
    def get_transition_call_opening_body(self, model: Transition) -> str:
        result = super().get_transition_call_opening_body(model)
        return self.join_with_strip([
                result,
                f"logger.info(\"T.O {self.current_class.name} "
                f"{self.current_state_machine.name} {model.id} "
                f"{model.source} {model.target}\");"
            ])

    def get_transition_call_success_closing_body(self, model: Transition) -> str:
        result = super().get_transition_call_success_closing_body(model)
        return self.join_with_strip([
                result,
                f"logger.info(\"T.CS {self.current_class.name} "
                f"{self.current_state_machine.name} {model.id} "
                f"{model.source} {model.target}\");"
            ])

    def get_transition_call_failure_closing_body(self, model: Transition) -> str:
        result = super().get_transition_call_failure_closing_body(model)
        return self.join_with_strip([
                result,
                f"logger.info(\"T.CF {self.current_class.name} "
                f"{self.current_state_machine.name} {model.id} "
                f"{model.source} {model.target}\");"
            ])

    def get_decision_structure_opening_body(self, model: StateMachine, state: State) -> str:
        result = super().get_decision_structure_opening_body(model, state)
        return self.join_with_strip([
                result,
                f"logger.info(\"D.O {self.current_class.name} "
                f"{self.current_state_machine.name} {state}\");"
            ])

    def get_decision_structure_closing_body(self, model: StateMachine, state: State) -> str:
        result = super().get_decision_structure_closing_body(model, state)
        return self.join_with_strip([
                result,
                f"logger.info(\"D.CF {self.current_class.name} "
                f"{self.current_state_machine.name} {state}\");"
            ])

    # LOGGER INITIALIZATION.
    def get_model_support_variables(self, model: SlcoModel) -> List[str]:
        # Add the logger initializer to the model.
        result = super().get_model_support_variables(model)
        result.append(
            self.logger_variable_and_static_initialization_template.render(
                model_name=model.name
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
