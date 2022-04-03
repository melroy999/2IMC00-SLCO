from __future__ import annotations

import json
import math
from typing import List, Dict, Tuple

import settings
from objects.ast.models import SlcoModel, Transition, StateMachine, State
from rendering.java.renderer import JavaModelRenderer
from rendering.measurements.renderer import MeasurementsModelRenderer


class CountMeasurementsModelRenderer(MeasurementsModelRenderer):
    """
    Create a subclass of the Java model renderer that renders increment statements for gathering performance metrics.
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
    def render_measurement_statement(self, model: str) -> str:
        """Render a statement that performs the measurement action."""
        # Use underscore instead of dot.
        model = model.replace(".", "_")
        return f"{model}++;"

    # INCLUDE VARIABLES.
    def render_state_machine_variable_declarations(self, model: StateMachine) -> str:
        # Add the variables required to count events.
        result = super().render_state_machine_variable_declarations(model)

        declarations = [
            result,
            "",
            "// Add variables needed for measurements."
        ]
        for state in model.states:
            ds_identifier = self.get_decision_structure_identifier(state)
            declarations.append(f"private long {ds_identifier}_O;")
            declarations.append(f"private long {ds_identifier}_F;")
            declarations.append(f"private long {ds_identifier}_S;")
            for t in model.state_to_transitions[state]:
                t_identifier = self.get_transition_identifier(t)
                declarations.append(f"private long {t_identifier}_O;")
                declarations.append(f"private long {t_identifier}_F;")
                declarations.append(f"private long {t_identifier}_S;")
        return "\n".join(declarations)

    def render_state_machine_count_report(self, model: StateMachine) -> str:
        """Report all of the count data gathered during the run."""
        statements = ["// Report all counts."]
        for state in model.states:
            ds_identifier = self.get_decision_structure_identifier(state)
            statements.append(f"logger.info(\"{ds_identifier}.O \" + {ds_identifier}_O);")
            statements.append(f"logger.info(\"{ds_identifier}.F \" + {ds_identifier}_F);")
            statements.append(f"logger.info(\"{ds_identifier}.S \" + {ds_identifier}_S);")
            for t in model.state_to_transitions[state]:
                t_identifier = self.get_transition_identifier(t)
                statements.append(f"logger.info(\"{t_identifier}.O \" + {t_identifier}_O);")
                statements.append(f"logger.info(\"{t_identifier}.F \" + {t_identifier}_F);")
                statements.append(f"logger.info(\"{t_identifier}.S \" + {t_identifier}_S);")
        return "\n".join(statements)

    def render_state_machine_post_execution(self, model: StateMachine) -> str:
        result = super().render_state_machine_post_execution(model)
        return "\n".join([result, self.render_state_machine_count_report(model)])

    # LOGGER INITIALIZATION.
    def get_model_support_variables(self, model: SlcoModel) -> List[str]:
        # Add the logger initializer to the model.
        result = super().get_model_support_variables(model)
        result.append(
            self.logger_variable_and_static_initialization_template.render(
                model_name=model.name,
                log_settings=settings.settings_abbreviations,
                log_file_size=settings.log_file_size,
                log_buffer_size=settings.log_buffer_size,
                compression_level=settings.compression_level,
                log_type="counting"
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

    def get_main_supportive_closing_method_calls(self, model: SlcoModel) -> List[str]:
        result = super().get_main_supportive_closing_method_calls(model)
        return result + [
            "// Include information about the model.",
            f"logger.info(\"JSON {self.get_model_information(model)}\");",
        ]

    def render_model(self, model: SlcoModel) -> str:
        # Add the data needed to abbreviate object names.
        self.add_support_data(model)
        return super().render_model(model)





