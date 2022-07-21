from __future__ import annotations

import logging
from typing import List, Union, Dict, Set

from z3 import z3

import settings
from objects.ast.models import DecisionNode, Transition, Variable

# Create solver objects for optimization problems.
s = z3.Optimize()


def optimize_lock_ordering_smt_full(decisions: List[Union[Transition, DecisionNode]]):
    """
    Re-order the transitions in an attempt to preemptively minimize the number of location sensitivity conflicts.
    """
    # FIXME: This code remains unused due to it being too slow in practice.
    # Save the current state of the solver.
    s.push()

    # Give each of the decisions a unique id.
    decision_indices: Dict[Union[Transition, DecisionNode]] = {
        d: i for i, d in enumerate(decisions)
    }

    # Find all the variables used within the decision node.
    used_variables: Set[Variable] = set()
    d: Union[Transition, DecisionNode]
    for d in decisions:
        used_variables.update(d.used_variables)
    used_variables: List[Variable] = list(used_variables)

    # Give each of the variables a unique id.
    variable_indices: Dict[Variable, int] = {
        v: i for i, v in enumerate(used_variables)
    }

    # # Create a table in which the use of variables is recorded through a boolean value.
    variable_use_table = z3.Array("U", z3.IntSort(), z3.ArraySort(z3.IntSort(), z3.BoolSort()))
    for d, i in decision_indices.items():
        for v, j in variable_indices.items():
            s.add(variable_use_table[i][j] == (v in d.used_variables))

    # Create a table that indicates which transition is placed at each slot of the aggregate table.
    index_table = z3.Array("I", z3.IntSort(), z3.IntSort())
    for i, _ in enumerate(decisions):
        s.add(0 <= index_table[i])
        s.add(index_table[i] < len(decisions))
        for j, _ in enumerate(decisions):
            if i != j:
                s.add(index_table[i] != index_table[j])

    # Create a table in which violations of the location sensitivity can be detected by looking at the predecessor.
    variable_aggregate_table = z3.Array("A", z3.IntSort(), z3.ArraySort(z3.IntSort(), z3.BoolSort()))
    for i, _ in enumerate(decisions):
        if i > 0:
            for j, _ in enumerate(used_variables):
                s.add(variable_aggregate_table[i][j] == z3.Or(
                    variable_use_table[index_table[i]][j],
                    variable_aggregate_table[i - 1][j]
                ))
        else:
            for j, _ in enumerate(used_variables):
                s.add(variable_aggregate_table[i][j] == variable_use_table[index_table[i]][j])

    # Create support variables that are set to one if a location sensitivity violation is created.
    violation_table = z3.Array("V", z3.IntSort(), z3.IntSort())
    j = 0
    for d, i in decision_indices.items():
        if i > 0:
            location_sensitive_variables = set(
                x.ref.var for x in d.location_sensitive_locks if not x.unavoidable_location_conflict
            )
            for v in location_sensitive_variables:
                s.add(z3.Implies(
                    variable_aggregate_table[index_table[i] - 1][variable_indices[v]], violation_table[j] == 1
                ))
                s.add(violation_table[j] >= 0)
                j += 1

    # The number of location sensitive lock violations needs to be minimized.
    # TODO: minimize the distance shift such that transitions stay in their original ordering as much as possible.
    if j > 0:
        s.minimize(
            len(decisions)**2 * sum(violation_table[i] for i in range(j))
        )

    print(s.check())
    model = s.model()
    print("positions")
    for i, _ in enumerate(decisions):
        print(i, model.evaluate(index_table[i], model_completion=True))
    print("violations")
    for i in range(j):
        print(i, model.evaluate(violation_table[i], model_completion=True))
    print("table")
    print(used_variables)
    for i, _ in enumerate(decisions):
        print([
                model.evaluate(
                    variable_aggregate_table[i][j], model_completion=True
                ) for j, _ in enumerate(used_variables)
        ])

    # Restore the state of the solver.
    s.pop()


def optimize_lock_ordering_smt_partial(node: DecisionNode) -> None:
    """Optimize the ordering of transitions to minimize the time locks remain active."""
    # FIXME: This code remains unused due to it being too slow in practice. One of two attempted solutions.
    # This solution does not consider potential lock ordering violations or original positions of transitions.
    # Save the current state of the solver.
    s.push()

    # Get the variables used within the node and count the number of objects within using the variable in question.
    used_variables = list(node.used_variables)
    nr_of_variables_uses = {
        v: 0 for v in used_variables
    }
    for o in node.decisions:
        for v in o.used_variables:
            nr_of_variables_uses[v] += 1

    # Create an array of indices that will dictate the order of the transitions.
    index_table = z3.Array("index_table", z3.IntSort(), z3.IntSort())

    # Ensure that all entries in the index table are unique and within range.
    n = len(node.decisions)
    for i in range(n):
        s.add(index_table[i] >= 0)
        s.add(index_table[i] < n)
        for j in range(n):
            if i < j:
                s.add(index_table[i] != index_table[j])

    # # Create a table in which the use of variables is recorded through a 0 or 1 value.
    variable_presence = z3.Array("variable_presence", z3.IntSort(), z3.ArraySort(z3.IntSort(), z3.IntSort()))
    for i, d in enumerate(node.decisions):
        for j, v in enumerate(used_variables):
            s.add(variable_presence[i][j] == (1 if v in d.used_variables else 0))

    # Create a two-dimensional table that counts the number of occurrences of a variable up to a certain position.
    # Create a void first row to simplify the added restrictions.
    variable_count_table = z3.Array("variable_count_table", z3.IntSort(), z3.ArraySort(z3.IntSort(), z3.IntSort()))
    for i in range(n + 1):
        for j, v in enumerate(used_variables):
            if i > 0:
                # Increment the value in the previous row if the variable in question is present in the target object.
                s.add(
                    variable_count_table[i][j] == variable_count_table[i-1][j] + variable_presence[index_table[i-1]][j]
                )
            else:
                # Set the cell of the first row to one if the variable is present, zero otherwise.
                s.add(variable_count_table[i][j] == 0)

    # Create arrays of lower and upper bounds for the active regions of each variable.
    lower_bounds = z3.Array("lower_bounds", z3.IntSort(), z3.IntSort())
    upper_bounds = z3.Array("upper_bounds", z3.IntSort(), z3.IntSort())
    for i, v in enumerate(used_variables):
        s.add(lower_bounds[i] >= 0)
        s.add(lower_bounds[i] < n + 1)
        s.add(variable_count_table[lower_bounds[i]][i] == 0)
        s.add(upper_bounds[i] >= 0)
        s.add(upper_bounds[i] < n + 1)
        s.add(variable_count_table[upper_bounds[i]][i] == nr_of_variables_uses[v])

    # Minimize the distance between the upper and lower bounds.
    s.minimize(sum(upper_bounds[i] - lower_bounds[i] for i, _ in enumerate(used_variables)))

    # Check and print solution.
    print(s.check())
    model = s.model()
    print("positions")
    for i in range(n):
        print(i, model.evaluate(index_table[i], model_completion=True))
    print("table")
    print([v.name for v in used_variables])
    for i in range(n + 1):
        print([
                model.evaluate(
                    variable_count_table[i][j], model_completion=True
                ) for j, _ in enumerate(used_variables)
        ])

    # Restore the state of the solver.
    s.pop()


def get_heuristic_cost(o: Union[Transition, DecisionNode], active_variables: Set[Variable]) -> float:
    """Get a heuristic score for the given object."""
    # Use heuristics to find and select an object that is deemed the best follow-up.
    # Calculate the differences and base the score on these differences.
    #   - -1 for every variable that is already active.
    #   - +1 for every variable that is active but not part of the transition.
    #   - +1 for every variable that is used by the transition but not active.
    #   - -1.1 for every not-active variable that is part of a uncontested location sensitivity lock.
    common_variables_score = -len(o.used_variables.intersection(active_variables))
    new_variables_score = len(o.used_variables.difference(active_variables))
    missing_variables_score = len(active_variables.difference(o.used_variables))

    # Add a bonus decrement if the transition has yet unopened location sensitive lock variables.
    modifier = len(set(
        i.ref.var for i in o.location_sensitive_locks if not (
                i.ref.var in active_variables or i.unavoidable_location_conflict
        )
    ))

    return common_variables_score + new_variables_score + missing_variables_score - 1.1 * modifier


def add_grouping_dependant_unavoidable_location_conflict_marks(node: DecisionNode) -> None:
    """
    Add unavoidable location conflict marks of transitions within the given node caused by multiple transitions having
    the same variables as a target in location sensitive locks.
    """
    # Gather all location sensitive locks together with their variables and source.
    variable_to_location_sensitive_transitions: Dict[Variable, Set[int]] = {v: set() for v in node.used_variables}
    for i, o in enumerate(node.decisions):
        for x in o.location_sensitive_locks:
            variable_to_location_sensitive_transitions[x.ref.var].add(i)

    # Having two or more transitions share location sensitive lock variable targets will always lead to conflicts.
    for v, transition_ids in variable_to_location_sensitive_transitions.items():
        if len(transition_ids) > 1:
            # Mark all location sensitive locks using v in the transition as unavoidable conflicts.
            for i in transition_ids:
                d = node.decisions[i]
                for x in d.location_sensitive_locks:
                    if x.ref.var == v and not x.unavoidable_location_conflict:
                        x.unavoidable_location_conflict = True
                        logging.info(
                            f"Flagging lock {x} in '{d}' due to the occurrence of an unavoidable location "
                            f"sensitivity violation caused by the grouping of the transitions."
                        )


def optimize_lock_ordering_greedy(node: DecisionNode) -> None:
    """
    Reorder the decisions within the decision node in an attempt to improve the performance of the locking mechanism.
    """
    # Order optimization is only required for deterministic or atomic sequential structures.
    if not node.is_deterministic:
        return

    # Ensure that additional unavoidable location conflict situations are detected prior to creating the ordering.
    add_grouping_dependant_unavoidable_location_conflict_marks(node)

    # Track which transitions use which variables.
    used_variables = list(node.used_variables)
    variable_to_transitions: Dict[Variable, Set[int]] = {
        v: set() for v in used_variables
    }
    for i, d in enumerate(node.decisions):
        for v in d.used_variables:
            variable_to_transitions[v].add(i)

    # Track which variables are currently active.
    active_variables: Set[Variable] = set()

    # Ideally a variable is opened by a object with location sensitive locks to prevent unavoidable strict unpacking.
    remaining_objects = list((d, i) for i, d in enumerate(node.decisions))
    reordered_objects = []
    while len(remaining_objects) > 0:
        # Sort according to the heuristic and pick the lowest valued element.
        # Draws are resolved by looking at the original ordering.
        remaining_objects.sort(key=lambda e: (e[0].priority, get_heuristic_cost(e[0], active_variables), e[1]))
        target_object, i = remaining_objects.pop(0)
        reordered_objects.append(target_object)

        # Update the unavoidable location conflict tags depending on which variables are active.
        for x in target_object.location_sensitive_locks:
            if x.ref.var in active_variables and not x.unavoidable_location_conflict:
                x.unavoidable_location_conflict = True
                logging.info(
                    f"Flagging lock {x} in '{target_object}' due to the occurrence of an unavoidable location "
                    f"sensitivity violation caused by the chosen order of the decisions."
                )

        # Update the active variables set based on the variables active within the extracted object.
        active_variables.update(target_object.used_variables)

        # Remove the transition from the variable-to-transition dictionary.
        for v, entries in variable_to_transitions.items():
            entries.discard(i)

            # Remove the variable from the active list if there are no remaining transitions left using it.
            if len(entries) == 0:
                active_variables.discard(v)

    # Set the new order.
    node.decisions = reordered_objects
