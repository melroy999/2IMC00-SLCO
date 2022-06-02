from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Union, Set, Dict

from z3 import z3

import settings
from objects.ast.models import DecisionNode, Transition
from smt.solutions.greedy.recursion import select_non_deterministic_node

if TYPE_CHECKING:
    from objects.ast.models import Variable

# Create solver objects for optimization problems.
s = z3.Optimize()
s2 = z3.Optimize()


def create_and_truth_table(transitions, alias_variables) -> None:
    """Create a truth table for the AND operation between the given list of transitions."""
    # Create the truth table.
    for t1 in transitions:
        for t2 in transitions:
            alias_variables[f"and{t1.id}_{t2.id}"] = z3.Bool(f"and{t1.id}_{t2.id}")

    for t1 in transitions:
        for t2 in transitions:
            if t2.id > t1.id:
                break

            # Add a new AND relation to the solver.
            e1 = t1.guard.smt
            e2 = t2.guard.smt

            # Existential quantifiers are not supported, so do the check separately.
            s.push()
            s.add(z3.And(e1, e2))
            # noinspection DuplicatedCode
            result = s.check().r == z3.Z3_L_TRUE
            s.pop()

            # Add the constant to the solver.
            s.add(alias_variables[f"and{t1.id}_{t2.id}"] == result)
            if t2.id < t1.id:
                s.add(alias_variables[f"and{t2.id}_{t1.id}"] == alias_variables[f"and{t1.id}_{t2.id}"])


def create_is_equal_table(transitions, alias_variables) -> None:
    """Create a truth table for the negated equality operation between the given list of transitions."""
    # Create the truth table.
    for t1 in transitions:
        for t2 in transitions:
            alias_variables[f"ieq{t1.id}_{t2.id}"] = z3.Bool(f"ieq{t1.id}_{t2.id}")

    for t1 in transitions:
        for t2 in transitions:
            if t2.id > t1.id:
                break

            # Add a new negated equality relation to the solver.
            e1 = t1.guard.smt
            e2 = t2.guard.smt

            # Existential quantifiers are not supported, so do the check separately.
            s.push()
            s.add(z3.Not(e1 == e2))
            # noinspection DuplicatedCode
            result = s.check().r == z3.Z3_L_FALSE
            s.pop()

            # Add the constant to the solver.
            s.add(alias_variables[f"ieq{t1.id}_{t2.id}"] == result)
            if t2.id < t1.id:
                s.add(alias_variables[f"ieq{t2.id}_{t1.id}"] == alias_variables[f"ieq{t1.id}_{t2.id}"])


def optimize_lock_ordering_smt_full(decisions: List[Union[Transition, DecisionNode]]):
    """
    Re-order the transitions in an attempt to preemptively minimize the number of location sensitivity conflicts.
    """
    # FIXME: This code remains unused due to it being too slow in practice. One of two attempted solutions.
    #   - This solution attempts to keep the transitions in the original order.
    #   - Moreover, it attempts to minimize the potential number of lock position sensitivity violations.
    # Save the current state of the solver.
    s2.push()

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
            s2.add(variable_use_table[i][j] == (v in d.used_variables))

    # Create a table that indicates which transition is placed at each slot of the aggregate table.
    index_table = z3.Array("I", z3.IntSort(), z3.IntSort())
    for i, _ in enumerate(decisions):
        s2.add(0 <= index_table[i])
        s2.add(index_table[i] < len(decisions))
        for j, _ in enumerate(decisions):
            if i != j:
                s2.add(index_table[i] != index_table[j])

    # Create a table in which violations of the location sensitivity can be detected by looking at the predecessor.
    variable_aggregate_table = z3.Array("A", z3.IntSort(), z3.ArraySort(z3.IntSort(), z3.BoolSort()))
    for i, _ in enumerate(decisions):
        if i > 0:
            for j, _ in enumerate(used_variables):
                s2.add(variable_aggregate_table[i][j] == z3.Or(
                    variable_use_table[index_table[i]][j],
                    variable_aggregate_table[i - 1][j]
                ))
        else:
            for j, _ in enumerate(used_variables):
                s2.add(variable_aggregate_table[i][j] == variable_use_table[index_table[i]][j])

    # Create support variables that are set to one if a location sensitivity violation is created.
    violation_table = z3.Array("V", z3.IntSort(), z3.IntSort())
    j = 0
    for d, i in decision_indices.items():
        if i > 0:
            location_sensitive_variables = set(
                x.ref.var for x in d.location_sensitive_locks if not x.unavoidable_location_conflict
            )
            for v in location_sensitive_variables:
                s2.add(z3.Implies(
                    variable_aggregate_table[index_table[i] - 1][variable_indices[v]], violation_table[j] == 1
                ))
                s2.add(violation_table[j] >= 0)
                j += 1

    # The number of location sensitive lock violations needs to be minimized.
    # Moreover, minimize the distance shift such that transitions stay in their original ordering as much as possible.
    if j > 0:
        s2.minimize(
            len(decisions)**2 * sum(violation_table[i] for i in range(j))
        )

    print(s2.check())
    model = s2.model()
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
    s2.pop()


def optimize_lock_ordering_smt_partial(node: DecisionNode) -> None:
    """Optimize the ordering of transitions to minimize the time locks remain active."""
    # FIXME: This code remains unused due to it being too slow in practice. One of two attempted solutions.
    #   - This solution does not consider potential lock ordering violations or original positions of transitions.
    # Save the current state of the solver.
    s2.push()

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
        s2.add(index_table[i] >= 0)
        s2.add(index_table[i] < n)
        for j in range(n):
            if i < j:
                s2.add(index_table[i] != index_table[j])

    # # Create a table in which the use of variables is recorded through a 0 or 1 value.
    variable_presence = z3.Array("variable_presence", z3.IntSort(), z3.ArraySort(z3.IntSort(), z3.IntSort()))
    for i, d in enumerate(node.decisions):
        for j, v in enumerate(used_variables):
            s2.add(variable_presence[i][j] == (1 if v in d.used_variables else 0))

    # Create a two-dimensional table that counts the number of occurrences of a variable up to a certain position.
    # Create a void first row to simplify the added restrictions.
    variable_count_table = z3.Array("variable_count_table", z3.IntSort(), z3.ArraySort(z3.IntSort(), z3.IntSort()))
    for i in range(n + 1):
        for j, v in enumerate(used_variables):
            if i > 0:
                # Increment the value in the previous row if the variable in question is present in the target object.
                s2.add(
                    variable_count_table[i][j] == variable_count_table[i-1][j] + variable_presence[index_table[i-1]][j]
                )
            else:
                # Set the cell of the first row to one if the variable is present, zero otherwise.
                s2.add(variable_count_table[i][j] == 0)

    # Create arrays of lower and upper bounds for the active regions of each variable.
    lower_bounds = z3.Array("lower_bounds", z3.IntSort(), z3.IntSort())
    upper_bounds = z3.Array("upper_bounds", z3.IntSort(), z3.IntSort())
    for i, v in enumerate(used_variables):
        s2.add(lower_bounds[i] >= 0)
        s2.add(lower_bounds[i] < n + 1)
        s2.add(variable_count_table[lower_bounds[i]][i] == 0)
        s2.add(upper_bounds[i] >= 0)
        s2.add(upper_bounds[i] < n + 1)
        s2.add(variable_count_table[upper_bounds[i]][i] == nr_of_variables_uses[v])

    # Minimize the distance between the upper and lower bounds.
    s2.minimize(sum(upper_bounds[i] - lower_bounds[i] for i, _ in enumerate(used_variables)))

    # Check and print solution.
    print(s2.check())
    model = s2.model()
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
    s2.pop()


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
    if not (node.is_deterministic or settings.atomic_sequential):
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


def create_smt_support_variables(remaining_transitions: List[Transition]) -> Dict[str, z3.ArithRef]:
    """Create a dictionary of SMT variables that are used in all implemented solutions."""
    # Create variables for each transition that will indicate the number of the group they are in.
    alias_variables = {f"g{t.id}": z3.Int(f"g{t.id}") for t in remaining_transitions}

    # Create support variables for the priorities assigned to the transitions.
    for t in remaining_transitions:
        v = alias_variables[f"p{t.id}"] = z3.Int(f"p{t.id}")
        s.add(v == t.priority)

    # Create the appropriate truth tables for the target transitions.
    create_and_truth_table(remaining_transitions, alias_variables)
    create_is_equal_table(remaining_transitions, alias_variables)

    return alias_variables


def insert_deterministic_group(
        alias_variables: Dict[str, z3.ArithRef],
        grouped_transitions: List[Transition],
        model: z3.ModelRef,
        non_deterministic_choices: List[Union[DecisionNode, Transition]]
) -> None:
    """Convert the given list of grouped transitions to a deterministic group."""
    # Ensure that duplicates in the group are handled appropriately.
    distinct_guard_groups: List[List[Transition]] = []
    for t in grouped_transitions:
        d: List[Transition]
        # Warning: syntax is correct. The else will only be executed if the loop isn't stopped prematurely.
        for d in distinct_guard_groups:
            if model.evaluate(alias_variables[f"ieq{t.id}_{d[0].id}"], model_completion=True):
                d.append(t)
                break
        else:
            distinct_guard_groups.append([t])
    # Sort the guard groups by priority and index and make non-deterministic groups if appropriate.
    deterministic_choices: List[Union[DecisionNode, Transition]] = []
    for d in distinct_guard_groups:
        d.sort(key=lambda x: (x.priority, x.id))
        if len(d) > 1:
            if not settings.use_random_pick:
                # Given that all transitions have the same guard, only the first will be reachable. Exclude others.
                decision_node = DecisionNode(False, d[:1], d[1:])
            else:
                # Create a non-deterministic block.
                decision_node = DecisionNode(False, d, [])
            optimize_lock_ordering_greedy(decision_node)
            deterministic_choices.append(decision_node)
        else:
            deterministic_choices.append(d[0])
    # Create a deterministic decision node for the current grouping and add it to the list of groupings.
    if len(deterministic_choices) == 1:
        non_deterministic_choices.append(deterministic_choices[0])
    else:
        # Sort the decisions based on the priority.
        deterministic_choices.sort(key=lambda x: (x.priority, x.id))
        decision_node = DecisionNode(True, deterministic_choices, [])
        optimize_lock_ordering_greedy(decision_node)
        non_deterministic_choices.append(decision_node)


def create_deterministic_decision_structures(remaining_transitions: List[Transition]) -> List[Transition, DecisionNode]:
    """Find and create deterministic structures for the provided list of transitions."""
    # Save the current state of the solver.
    s.push()

    # Create all common variables needed within the SMT solution.
    alias_variables = create_smt_support_variables(remaining_transitions)

    # Ensure that the part of the group variable of each transition can only be zero or one.
    for t in remaining_transitions:
        v = alias_variables[f"g{t.id}"]
        s.add(z3.And(v >= 0, v < 2))

    # Calculate truth matrices for the transitions that still need to be assigned to a group.
    non_deterministic_choices: List[Union[DecisionNode, Transition]] = []
    while len(remaining_transitions) > 0:
        # Save the current state of the solver.
        s.push()

        # Create the rules needed to construct valid groupings.
        for t1 in remaining_transitions:
            v = alias_variables[f"g{t1.id}"]

            # It must hold that the transition has no overlap with any of the members in the same group.
            inner_or = z3.Or([
                z3.And(
                    alias_variables[f"g{t2.id}"] == 1,
                    alias_variables[f"and{t1.id}_{t2.id}"],
                    z3.Not(alias_variables[f"ieq{t1.id}_{t2.id}"])
                ) for t2 in remaining_transitions if t1.id != t2.id
            ])
            s.add(z3.Implies(v == 1, z3.Not(inner_or)))

            # It must hold that the priority of each member in the group is the same.
            inner_or = z3.Or([
                z3.And(
                    alias_variables[f"g{t2.id}"] == 1,
                    alias_variables[f"p{t2.id}"] != alias_variables[f"p{t1.id}"]
                ) for t2 in remaining_transitions if t1.id != t2.id
            ])
            s.add(z3.Implies(v == 1, z3.Not(inner_or)))

        # Maximize the size of the group.
        s.maximize(sum(alias_variables[f"g{t.id}"] for t in remaining_transitions))

        # Get the model solution and extract the selected transitions.
        result, model = s.check(), s.model()
        if result.r == z3.Z3_L_UNDEF:
            print(result, model)
            raise Exception("Unknown result.")
        if result.r == z3.Z3_L_FALSE:
            print(result, model)
            raise Exception("Unsatisfiable result.")

        # Find the target transitions that are part of the deterministic group and process them accordingly.
        grouped_transitions: List[Transition] = [
            t for t in remaining_transitions if model.evaluate(alias_variables[f"g{t.id}"], model_completion=True) == 1
        ]

        # Remove the transitions from the to be processed list.
        for t in grouped_transitions:
            remaining_transitions.remove(t)

        # Insert a deterministic group into the list of non-deterministic choices containing the selected transitions.
        insert_deterministic_group(alias_variables, grouped_transitions, model, non_deterministic_choices)

        # Restore the state of the solver.
        s.pop()

    # Restore the state of the solver.
    s.pop()
    return non_deterministic_choices


# FIXME: Old, too slow and unmaintained code that could be useful still for the document.
#   - Code has been rewritten, but it has not been fully tested. The performance has somewhat increased.
def create_deterministic_decision_structures_full_smt(
    remaining_transitions: List[Transition]
) -> List[Transition, DecisionNode]:
    """
    Use z3 optimization to create a minimally sized list of groups of transitions in which the guard statements'
    active regions do not overlap.
    """
    # No elements to process if an empty list is given.
    if len(remaining_transitions) == 0:
        return []

    # Give all of the transitions within the list an unique ID.
    # This id can be used to find the original ordering.
    for i, t in enumerate(remaining_transitions):
        t.id = i

    # Save the current state of the solver.
    s.push()

    # Create all common variables needed within the SMT solution.
    alias_variables = create_smt_support_variables(remaining_transitions)

    # Put bounds on the group variables such that they are always within a range between zero and the list's size.
    for t in remaining_transitions:
        v = alias_variables[f"g{t.id}"]
        s.add(z3.And(v >= 0, v < len(remaining_transitions)))

    # Ensure that a transition can only be assigned to a group if it does not overlap with others in the same group.
    # However, make the following exception: allow transitions to be in the same group if they are exactly equal.
    for t1 in remaining_transitions:
        v = alias_variables[f"g{t1.id}"]

        for k in range(len(remaining_transitions)):
            # It must hold that the transition has no overlap with any of the members in the same group.
            inner_or = z3.Or([
                z3.And(
                    alias_variables[f"g{t2.id}"] == k,
                    alias_variables[f"and{t1.id}_{t2.id}"],
                    z3.Not(alias_variables[f"ieq{t1.id}_{t2.id}"])
                ) for t2 in remaining_transitions if t1.id != t2.id
            ])
            s.add(z3.Implies(v == k, z3.Not(inner_or)))

            # It must hold that the priority of each member in the group is the same.
            inner_or = z3.Or([
                z3.And(
                    alias_variables[f"g{t2.id}"] == k,
                    alias_variables[f"p{t2.id}"] != alias_variables[f"p{t1.id}"]
                ) for t2 in remaining_transitions if t1.id != t2.id
            ])
            s.add(z3.Implies(v == k, z3.Not(inner_or)))

    # Minimize the sum of all groups such that the least number of groups are generated with maximum size.
    s.minimize(sum(alias_variables[f"g{t.id}"] for t in remaining_transitions))

    # Get the model solution and extract the selected transitions.
    result, model = s.check(), s.model()
    if result.r == z3.Z3_L_UNDEF:
        print(result, model)
        raise Exception("Unknown result.")
    if result.r == z3.Z3_L_FALSE:
        print(result, model)
        raise Exception("Unsatisfiable result.")

    # Find the groupings. Elements within the same group assigned should be added to the same group.
    non_deterministic_choices: List[Union[DecisionNode, Transition]] = []
    for k, _ in enumerate(remaining_transitions):
        grouped_transitions: List[Transition] = []
        for t in remaining_transitions:
            if model.evaluate(alias_variables[f"g{t.id}"], model_completion=True) == k:
                grouped_transitions.append(t)

        # Skip if no transitions.
        if len(grouped_transitions) == 0:
            continue

        # Insert a deterministic group into the list of non-deterministic choices containing the selected transitions.
        insert_deterministic_group(alias_variables, grouped_transitions, model, non_deterministic_choices)

    # Restore the state of the solver.
    s.pop()

    # Create and return a non-deterministic decision node for the given decisions.
    return non_deterministic_choices


def create_decision_groupings(transitions: List[Transition]) -> DecisionNode:
    """
    Use z3 optimization to create a minimally sized collections of groups of transitions in which the guard statements'
    active regions do not overlap.
    """
    # Give all transitions an unique identity.
    for i, t in enumerate(transitions):
        t.id = i

    # Filter out all transitions that are invariably true.
    trivially_satisfiable_transitions = []
    remaining_transitions = []
    excluded_transitions = []
    for t in transitions:
        if t.guard.is_true():
            trivially_satisfiable_transitions.append(t)
        elif t.guard.is_false():
            excluded_transitions.append(t)
        else:
            remaining_transitions.append(t)

    decision_node = select_non_deterministic_node(list(remaining_transitions))

    # if settings.no_deterministic_structures:
    #     # Do not create deterministic structures when the no determinism flag is provided.
    #     non_deterministic_choices: List[Transition, DecisionNode] = remaining_transitions
    # else:
    #     if settings.use_full_smt_dsc:
    #         non_deterministic_choices = create_deterministic_decision_structures_full_smt(remaining_transitions)
    #     else:
    #         non_deterministic_choices = create_deterministic_decision_structures(remaining_transitions)
    #
    # # Sort the decisions based on the priority.
    # non_deterministic_choices += trivially_satisfiable_transitions
    # non_deterministic_choices.sort(key=lambda x: (x.priority, x.id))
    #
    # # Create and return a non-deterministic decision node for the given decisions.
    # decision_node = DecisionNode(False, non_deterministic_choices, excluded_transitions)
    # optimize_lock_ordering_greedy(decision_node)
    return decision_node
