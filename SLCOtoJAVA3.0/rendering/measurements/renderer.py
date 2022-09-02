from __future__ import annotations

import json
import math
from typing import List, Dict, Tuple

import settings
from objects.ast.models import SlcoModel, Transition, StateMachine, State
from rendering.java.renderer import JavaModelRenderer


class MeasurementsModelRenderer(JavaModelRenderer):
    """
    Create a subclass of the Java model renderer that provides the required information to perform measurements.
    """

    def __init__(self):
        super().__init__()

        # Support variables.
        self.abbreviation_mapping: Dict[Tuple, str] = dict()

    # LOG STATEMENTS.
    def get_decision_structure_identifier(self, model: State):
        """Get a string with which decision structures can be identified in the logs."""
        return self.abbreviation_mapping[(self.current_class, self.current_state_machine, model)]

    def get_transition_identifier(self, model: Transition):
        """Get a string with which transitions can be identified in the logs."""
        return self.abbreviation_mapping[(self.current_class, self.current_state_machine, model)]

    def render_measurement_statement(self, model: str) -> str:
        """Render a statement that performs the measurement action."""
        return ""

    def get_transition_call_opening_body(self, model: Transition) -> str:
        result = super().get_transition_call_opening_body(model)
        return self.join_with_strip([
                result,
                self.render_measurement_statement(f"{self.get_transition_identifier(model)}.O")
            ])

    def get_transition_call_success_closing_body(self, model: Transition) -> str:
        result = super().get_transition_call_success_closing_body(model)
        return self.join_with_strip([
                result,
                self.render_measurement_statement(f"{self.get_transition_identifier(model)}.S"),
                self.render_measurement_statement(f"{self.get_decision_structure_identifier(model.source)}.S")
            ])

    def get_transition_call_failure_closing_body(self, model: Transition) -> str:
        result = super().get_transition_call_failure_closing_body(model)
        return result
        # TODO: Removed, since failure data can be derived from the other two entries. Furthermore, it creates an
        #  imbalance in the number of log actions each path through the program has to take.
        # return self.join_with_strip([
        #         result,
        #         self.render_measurement_statement(f"{self.get_transition_identifier(model)}.F")
        #     ])

    def get_decision_structure_opening_body(self, model: StateMachine, state: State) -> str:
        result = super().get_decision_structure_opening_body(model, state)
        return self.join_with_strip([
                result,
                self.render_measurement_statement(f"{self.get_decision_structure_identifier(state)}.O")
            ])

    def get_decision_structure_closing_body(self, model: StateMachine, state: State) -> str:
        result = super().get_decision_structure_closing_body(model, state)
        return result
        # TODO: Removed, since failure data can be derived from the other two entries. Furthermore, it creates an
        #  imbalance in the number of log actions each path through the program has to take.
        # return self.join_with_strip([
        #         result,
        #         self.render_measurement_statement(f"{self.get_decision_structure_identifier(state)}.F")
        #     ])

    def add_support_data(self, model: SlcoModel) -> None:
        """Add the data needed to render log messages in an uniform and equally impactful manner."""
        # The goal is to have all log messages within the model be of the same length, with the primary aim being that
        # all transitions should be impacted to an equal degree by the logging.
        transition_ids = dict()
        decision_structure_ids = dict()
        for c in model.classes:
            for sm in c.state_machines:
                for state in sm.states:
                    decision_structure_ids[(c, sm, state)] = f"{len(decision_structure_ids)}"
                    for t in sm.state_to_transitions[state]:
                        transition_ids[(c, sm, t)] = f"{len(transition_ids)}"

        # Ensure that all abbreviations are of equal length.
        length = max([len(v) for v in transition_ids.values()] + [len(v) for v in decision_structure_ids.values()])
        for key, value in transition_ids.items():
            self.abbreviation_mapping[key] = "T" + value.zfill(length)
        for key, value in decision_structure_ids.items():
            self.abbreviation_mapping[key] = "D" + value.zfill(length)

    def get_model_information(self, model: SlcoModel) -> str:
        """Represent the model as a json object."""
        # Create a structure of dictionaries and convert it to json.
        classes = dict()
        for c in model.classes:
            state_machines = dict()
            for sm in c.state_machines:
                decision_structures = dict()
                for state in sm.states:
                    transitions = dict()
                    for t in sm.state_to_transitions[state]:
                        transitions[t.id] = {
                            "name": str(t),
                            "id": self.abbreviation_mapping[(c, sm, t)],
                            "source": t.source.name,
                            "target": t.target.name,
                            "priority": t.priority,
                            "is_excluded": t.is_excluded
                        }
                    decision_structures[state.name] = {
                        "source": state.name,
                        "id": self.abbreviation_mapping[(c, sm, state)],
                        "transitions": transitions
                    }
                state_machines[sm.name] = {
                    "name": sm.name,
                    "states": [s.name for s in sm.states],
                    "decision_structures": decision_structures
                }
            classes[c.name] = {
                "name": c.name,
                "state_machines": state_machines
            }
        data = {
            "name": model.name,
            "settings": settings.original_arguments,
            "classes": classes
        }

        return json.dumps(data).replace("\"", "\\\"")

    def render_model(self, model: SlcoModel) -> str:
        # Add the data needed to abbreviate object names.
        self.add_support_data(model)
        return super().render_model(model)





