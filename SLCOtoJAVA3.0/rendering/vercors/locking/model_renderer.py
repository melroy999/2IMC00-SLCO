from typing import List, Dict, Set, Tuple

import networkx as nx

from libraries.slcolib import VariableRef
from objects.ast.interfaces import SlcoStatementNode, SlcoLockableNode
from objects.ast.models import Transition, StateMachine, State
from objects.ast.util import get_class_variable_references
from objects.locking.models import LockingNode, Lock
from rendering.vercors.model_renderer import VercorsModelRenderer


class VercorsLockingModelRenderer(VercorsModelRenderer):
    """
    Create a subclass of the java model renderer that renders VerCors verification statements to verify the correctness
    of the rendered models from a locking structure point of view.
    """

    def __init__(self):
        super().__init__()

        # Overwrite the locking instructions template to instead work with ids and assumptions.
        self.locking_instruction_template = self.env.get_template(
            "vercors/locking/locking_instruction.jinja2template"
        )

        # Add additional templates.
        self.lock_id_presence_contract_template = self.env.get_template(
            "vercors/locking/lock_id_presence_contract.jinja2template"
        )
        self.lock_id_absence_contract_template = self.env.get_template(
            "vercors/locking/lock_id_absence_contract.jinja2template"
        )
        self.vercors_locking_check_template = self.env.get_template(
            "vercors/locking/locking_check.jinja2template"
        )

    def render_locking_check(self, model: SlcoStatementNode) -> str:
        # Use a different template and render the original locks instead.
        # Create a list containing all of the lock requests associated with the node.
        lock_requests = set()
        for i in model.locking_atomic_node.entry_node.original_locks:
            lock_requests.update(i.lock_requests)
        lock_requests = sorted(lock_requests, key=lambda x: x.id)

        lock_request_ids = [i.id for i in lock_requests]
        lock_request_comments = [self.get_variable_ref_comment(i.ref) for i in lock_requests]
        lock_request_entries = list(zip(lock_request_ids, lock_request_comments))

        if len(lock_requests) == 0:
            return ""
        else:
            return self.vercors_locking_check_template.render(
                lock_request_entries=lock_request_entries
            )

    @staticmethod
    def create_locking_order(model: LockingNode) -> List[Lock]:
        """Create a locking order that ensures that no erroneous permission errors will pop up during verification."""
        # Link every lock to the associated variable reference.
        variable_ref_to_lock: Dict[VariableRef, Lock] = {i.original_ref: i for i in model.original_locks}
        if len(variable_ref_to_lock) != len(model.original_locks):
            raise Exception("Unexpected duplicate elements in original locks list.")

        # If a lock is used within another lock, then it needs to be precede the latter in the permission list.
        # Create a directed graph for this purpose.
        graph = nx.DiGraph()
        references: Set[VariableRef] = get_class_variable_references(model.partner)
        while len(references) > 0:
            target = references.pop()
            if target not in variable_ref_to_lock:
                # Skip locks not in the dictionary.
                continue
            graph.add_node(variable_ref_to_lock[target])
            if target.index is not None:
                sub_references: Set[VariableRef] = get_class_variable_references(target.index)
                for r in sub_references:
                    if r not in variable_ref_to_lock:
                        # Skip locks not in the dictionary.
                        continue
                    graph.add_edge(variable_ref_to_lock[target], variable_ref_to_lock[r])

        # Return the list in topological order.
        return list(nx.topological_sort(graph))

    def render_vercors_lock_id_presence_contract(self, model: SlcoLockableNode) -> str:
        """
        Render the VerCors statements that verify whether the locking structure is intact and behaving as expected.
        """
        # Find the lock requests present at the entry points and exit points of the given statement.
        entry_node_lock_requests = sorted(
            list(model.locking_atomic_node.entry_node.locking_instructions.requires_lock_requests), key=lambda x: x.id
        )
        success_exit_lock_requests = sorted(
            list(model.locking_atomic_node.success_exit.locking_instructions.ensures_lock_requests), key=lambda x: x.id
        )
        failure_exit_lock_requests = sorted(
            list(model.locking_atomic_node.failure_exit.locking_instructions.ensures_lock_requests), key=lambda x: x.id
        )

        # Create a disjunction of target ids for each of the nodes.
        entry_node_disjunction = " || ".join([f"_i == {i.id}" for i in entry_node_lock_requests])
        success_exit_disjunction = " || ".join([f"_i == {i.id}" for i in success_exit_lock_requests])
        failure_exit_disjunction = " || ".join([f"_i == {i.id}" for i in failure_exit_lock_requests])

        # Create human-readable comments for verification purposes.
        entry_node_comment_string = ", ".join([f"{i.id}: {i}" for i in entry_node_lock_requests])
        success_exit_comment_string = ", ".join([f"{i.id}: {i}" for i in success_exit_lock_requests])
        failure_exit_comment_string = ", ".join([f"{i.id}: {i}" for i in failure_exit_lock_requests])

        # Render the statement.
        return self.lock_id_presence_contract_template.render(
            entry_node_disjunction=entry_node_disjunction,
            success_exit_disjunction=success_exit_disjunction,
            failure_exit_disjunction=failure_exit_disjunction,
            entry_node_comment_string=entry_node_comment_string,
            success_exit_comment_string=success_exit_comment_string,
            failure_exit_comment_string=failure_exit_comment_string,
            target_locks_list_size=self.current_state_machine.target_locks_list_size
        )

    def render_vercors_expression_control_node_contract_requirements(
            self, model: SlcoStatementNode
    ) -> Tuple[str, List[str]]:
        # Add the lock request id check to the expression control node contract.
        requirements, pure_functions = super().render_vercors_expression_control_node_contract_requirements(model)
        lock_request_id_check = self.render_vercors_lock_id_presence_contract(model)
        return "\n\n".join(v.strip() for v in [requirements, lock_request_id_check] if v != ""), pure_functions

    def render_vercors_transition_contract_body(self, model: Transition) -> Tuple[str, List[str]]:
        # Add the lock request id check to the transition contract.
        result, pure_functions = super().render_vercors_transition_contract_body(model)
        lock_request_id_check = self.render_vercors_lock_id_presence_contract(model.guard)
        return "\n\n".join(v for v in [result, lock_request_id_check] if v != ""), pure_functions

    def render_vercors_decision_structure_lock_id_absence_contract(self) -> str:
        """
        Render the VerCors statements that verify that the decision structure starts and ends with no locks active.
        """
        # Render the statement.
        return self.lock_id_absence_contract_template.render(
            target_locks_list_size=self.current_state_machine.target_locks_list_size
        )

    def get_decision_structure_contract_body(self, model: StateMachine, state: State) -> Tuple[str, List[str]]:
        # Add the lock request id check to the decision node contract.
        result, pure_functions = super().get_decision_structure_contract_body(model, state)
        lock_request_id_check = self.render_vercors_decision_structure_lock_id_absence_contract()
        return "\n\n".join(v for v in [result, lock_request_id_check] if v != ""), pure_functions

    def render_state_machine_variable_declarations(self, model: StateMachine) -> str:
        # Add an instantiation for the lock_requests list.
        result = super().render_state_machine_variable_declarations(model)

        return "\n".join([
            result,
            "// A list of lock requests. A value of 1 denotes that the given target is locked, and 0 implies no lock.",
            "private final int[] lock_requests;"
        ])

    def render_state_machine_constructor_body(self, model: StateMachine) -> str:
        # Add an instantiation for the lock_requests list.
        result = super().render_state_machine_constructor_body(model)

        return "\n".join([
            result,
            "// Instantiate the lock requests array.",
            f"lock_requests = new int[{model.target_locks_list_size}];"
        ])


