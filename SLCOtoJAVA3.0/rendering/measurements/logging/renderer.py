from __future__ import annotations

import json
import math
from typing import List, Dict, Tuple

import settings
from objects.ast.models import SlcoModel, Transition, StateMachine, State
from rendering.java.renderer import JavaModelRenderer
from rendering.measurements.renderer import MeasurementsModelRenderer


class LogMeasurementsModelRenderer(MeasurementsModelRenderer):
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
    def render_measurement_statement(self, model: str) -> str:
        """Render a statement that performs the measurement action."""
        return f"logger.info(\"{model}\");"

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
                log_type="logging"
            )
        )
        return result

    def get_package_name(self) -> str:
        return ".".join(v for v in [super().get_package_name(), "logging"] if v != "")

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

    def get_main_supportive_closing_method_calls(self, model: SlcoModel) -> List[str]:
        result = super().get_main_supportive_closing_method_calls(model)
        return result + [
            "// Include information about the model.",
            f"logger.info(\"JSON {self.get_model_information(model)}\");",
            self.sleep_template.render(),
            self.force_rollover_template.render(),
            self.sleep_template.render()
        ]

    def render_model(self, model: SlcoModel) -> str:
        # Add the data needed to abbreviate object names.
        self.add_support_data(model)
        return super().render_model(model)





