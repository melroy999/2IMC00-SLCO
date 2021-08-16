from __future__ import annotations

import operator
from collections import Counter
from functools import reduce
from typing import Union, List

import objects.ast.models as models

# TODO: Simplify comparisons when negated by swapping the operator.
#   XOR breaks when doing this--it is broken still: x > 0 xor x <= 0 == true.
#   Tests need to be written for this situation... moreover, a solution needs to be found too...
# TODO: Write out power operations when possible to simplify division logic.
# TODO: Figure out why """E7: `x ** 2 % y` -> `x ** 2 % y`""" results in a SMT error. Lack of brackets? Power broken?
#   In smt, the power operator might only work with real values?
# TODO: Figure out why """E8(R5): `x * y % x` -> `0`""" results in an infinite equality loop.
#   This isn't a linear equation. Hence, it might be unresolvable.
# TODO: Write a merge procedure for multiplication and negated primaries with multiplications.
# TODO: Write a merge procedure for addition and negated primaries with additions.


# Conjunction
def simplify_conjunction(e: models.Expression):
    """
    Simplify the given conjunction, such that the following holds:
        - R1: All members evaluating to `true` are eliminated.
        - R2: Conjunctions with a false member are simplified to `false`.
        - R3: Duplicate members are removed.
        - R4: A single value should be returned if the simplified operator is over a single value.
        - R5: A conjunction over no elements is equivalent to `true`.
        - R6: A conjunction with two members that are each other's negation is simplified to `false`.

    Assumptions:
        - A1: All viable members have been simplified to `true` and `false` before evaluation.

    Examples:
        - E1(R1): `expr1 and expr2 and true` -> `expr1 and expr2`
        - E2(R3): `expr1 and expr1 and expr2` -> `expr1 and expr2`
        - E3(R2): `expr1 and expr2 and false` -> `false`
        - E4(R6): `expr1 and !(expr1)` -> `false`
        - E5(R1, R4): `true and true` -> `true`
        - E6(R1, R4): `expr1 and true` -> `true`
        - E7: `expr1 and expr2` -> `expr1 and expr2`
        - E8: `expr1` -> `expr1`
        - E9(R5): `[]` -> `true`

    :param e: An expression object with the conjunction operator.
    :return: A simplification of input parameter e.
    """
    basic_values = [v for v in e.values if isinstance(v, models.Primary) and v.value is not None]

    if any(not v.signed_value for v in basic_values):
        # R2: If any member is false, the expression result is false.
        return models.Primary(target=False)
    else:
        # R1, R3: Filter out all true values and duplicate values.
        e.values = list({v for v in e.values if not isinstance(v, models.Primary) or v.value is None})

    if len(e.values) == 0:
        # R5
        return models.Primary(target=True)
    elif len(e.values) == 1:
        # R4
        return e.values[0]
    elif e.is_false():
        # R6: The equality routine uses SMT. Hence, a direct false check is more efficient than a pair-wise check.
        return models.Primary(target=False)
    return e


# Disjunction
def simplify_disjunction(e: models.Expression):
    """
    Simplify the given disjunction, such that the following holds:
        - R1: All members evaluating to `false` are eliminated.
        - R2: Conjunctions with a true member are simplified to `true`.
        - R3: Duplicate members are removed.
        - R4: A single value should be returned if the simplified operator is over a single value.
        - R5: A disjunction over no elements is equivalent to `false`.
        - R6: A conjunction with two members that are each other's negation is simplified to `true`.

    Assumptions:
        - A1: All viable members have been simplified to `true` and `false` before evaluation.

    Examples:
        - E1(R1): `expr1 or expr2 or false` -> `expr1 or expr2`
        - E2(R3): `expr1 or expr1 or expr2` -> `expr1 or expr2`
        - E3(R2): `expr1 or expr2 or true` -> `true`
        - E4(R6): `expr1 or !(expr1)` -> `true`
        - E5(R1, R4): `false or false` -> `false`
        - E6(R1, R4): `expr1 or false` -> `expr1`
        - E7: `expr1 or expr2` -> `expr1 or expr2`
        - E8: `expr1` -> `expr1`
        - E9(R5): `[]` -> `false`

    :param e: An expression object with the disjunction operator.
    :return: A simplification of input parameter e.
    """
    basic_values = [v for v in e.values if isinstance(v, models.Primary) and v.value is not None]

    if any(v.signed_value for v in basic_values):
        # R2: If any member is true, the expression result is true.
        return models.Primary(target=True)
    else:
        # R1, R3: Filter out all false values and duplicate values.
        e.values = list({v for v in e.values if not isinstance(v, models.Primary) or v.value is None})

    if len(e.values) == 0:
        # R5
        return models.Primary(target=False)
    elif len(e.values) == 1:
        # R4
        return e.values[0]
    elif e.is_true():
        # R6: The equality routine uses SMT. Hence, a direct true check is more efficient than a pair-wise check.
        return models.Primary(target=True)
    return e


# Exclusive Disjunction
def simplify_exclusive_disjunction(e: models.Expression):
    """
    Simplify the given exclusive disjunction, such that the following holds:
        - R1: A constant is returned when both members are a constant value.
        - R2: Constant members are eliminated through rewriting using De Morgan's laws.
        - R3: Pairs of duplicate members are resolved to `false'.
        - R4: Pairs of members that are each other's negation are resolved to `true'.
        - R5: A single value should be returned if the simplified operator is over a single remaining value.
        - R6: An exclusive disjunction over no elements is equivalent to `false`.

    Assumptions:
        - A1: All viable members have been simplified to `true` and `false` before evaluation.

    Examples:
        - E1(R6): `expr1 xor expr2 xor false xor false` -> `expr1 xor expr2`
        - E2(R2): `expr1 xor true xor expr2 xor false` -> `expr1 xor !expr2`
        - E3(R2, R3): `expr1 xor expr1 xor expr1 xor true` -> `!expr1`
        - E4(R4, R5): `expr1 xor !expr1 xor expr1 xor true` -> `expr1`
        - E5(R4, R5): `!expr1 xor expr1 xor !expr1 xor true` -> `!expr1`
        - E6(R3, R5): `expr1 xor expr1 xor true` -> `true`
        - E7(R4, R5): `expr1 xor !expr1 xor true` -> `false`
        - E8: `expr1 xor expr2` -> `expr1 xor expr2`
        - E9: `expr1` -> `expr1`
        - E10(R6): `[]` -> `false`
        - E11(R3, R5): `expr1 xor expr1` -> `false`
        - E12(R4, R5): `expr1 xor !expr1` -> `true`
        - E13(R1): `true xor false` -> `true`

    :param e: An expression object with the exclusive disjunction operator.
    :return: A simplification of input parameter e.
    """
    basic_values = [v for v in e.values if isinstance(v, models.Primary) and v.value is not None]
    values = [v for v in e.values if not isinstance(v, models.Primary) or v.value is None]

    # Group the values by itself and its negation.
    grouping = dict()
    for v in values:
        # Add a 1 or -1 based on whether the value is a negation.
        if v in grouping:
            grouping[v] = grouping.get(v, []) + [1]
        else:
            inverse = next((k for k in grouping.keys() if k.is_negation_equivalent(v)), None)
            if inverse is not None:
                grouping[inverse] = grouping.get(inverse, []) + [-1]
            else:
                grouping[v] = grouping.get(v, []) + [1]

    # Simplify duplicates and negation pairs.
    processed_values = []
    for key, factors in grouping.items():
        if len(factors) % 2 == 1:
            factors.append(None)
        # Process the list in pairs of two.
        pairs = list(zip(factors[::2], factors[1::2]))
        for i, j in pairs:
            if j is None:
                if i == -1:
                    processed_values.append(
                        simplify_primary(models.Primary(sign="not", target=key), recursion=False)
                    )
                else:
                    processed_values.append(key)
            elif i + j == 0:
                basic_values.append(models.Primary(value=True))
            else:
                basic_values.append(models.Primary(value=False))

    # R2: Determine the constant member of the expression and rewrite the equation if possible.
    adjusted_value = None
    if len(basic_values) > 0:
        adjusted_value = reduce(operator.__xor__, [v.signed_value for v in basic_values])

    if adjusted_value:
        # A true value requires one of the other members to be negated. Default to true otherwise.
        if len(processed_values) == 0:
            processed_values.append(models.Primary(target=True))
        else:
            processed_values[-1] = simplify_primary(
                models.Primary(sign="not", target=processed_values[-1]), recursion=False
            )
    e.values = processed_values

    if len(e.values) == 0:
        # R7
        return models.Primary(target=False)
    elif len(e.values) == 1:
        # R5
        return e.values[0]
    else:
        return e


# Multiplication
def simplify_multiplication(e: models.Expression):
    """
    Simplify the given multiplication, such that the following holds:
        - R1: Only one constant member remains after simplification.
        - R2: The constant member, when present, is the first member of the expression.
        - R3: Multiplication by zero is simplified to zero.
        - R4: If a constant factor of one/minus one is present, it is eliminated.
        - R5: The multiplication is wrapped in a negation if the constant factor is negative.
        - R6: A single value should be returned if the simplified operator is over a single value.
        - R7: A multiplication over no elements is equivalent to `1`.
        - R8: The members within the multiplication are never negative.
        - TODO: R9: The multiplication of a number with a division is simplified when possible.
            (Procedure, examples and tests need to be implemented)

    Assumptions:
        - A1: All members have been simplified before evaluation.

    Examples:
        - E1(R5, R8): `-2 * expr1 * expr2` -> `-(2 * expr1 * expr2)`
        - E2(R4, R5): `-1 * expr1 * expr2` -> `-(expr1 * expr2)`
        - E3(R3): `0 * expr1 * expr2` -> `0`
        - E4(R4): `1 * expr1 * expr2` -> `expr1 * expr2`
        - E5(R1, R2): `2 * expr1 * expr2 * 4` -> `8 * expr1 * expr2`
        - E6(R5, R8): `expr1 * -expr2` -> `-(expr1 * expr2)`
        - E7(R2, R8): `expr1 * -expr2 * -2` -> `2 * expr1 * expr2`
        - E8(R4): `1 * expr1` -> `expr1`
        - E9(R4, R6): `-1 * expr1` -> `-(expr1)`
        - E10(R1, R2, R6): `-1 * 2` -> `-2`
        - E11: `expr1 * expr2` -> `expr1 * expr2`
        - E12: `expr1` -> `expr1`
        - E13(R7): `[]` -> `1`

    :param e: An expression object with the multiplication operator.
    :return: A simplification of input parameter e.
    """
    basic_values = [v for v in e.values if isinstance(v, models.Primary) and v.value is not None]
    adjusted_value = reduce(operator.__mul__, [v.signed_value for v in basic_values], 1)
    values = [v for v in e.values if not isinstance(v, models.Primary) or v.value is None]

    # R8: Reduce the number of negations by only having one negation as a wrapper and using all positive factors.
    nr_of_negations = len([v for v in values if isinstance(v, models.Primary) and v.sign == "-"])
    positive_factors = []
    for v in values:
        if isinstance(v, models.Primary) and v.sign == "-":
            positive_factors.append(models.Primary(target=v.body or v.ref or v.value))
        else:
            positive_factors.append(v)

    # Negate the primary factor if an odd number of negations is present.
    if nr_of_negations % 2 == 1:
        adjusted_value *= -1

    if len(values) > 0 and adjusted_value != 0:
        if adjusted_value < 0:
            if adjusted_value == -1:
                # R4: Leave out the constant factor.
                e.values = positive_factors
            else:
                # R1, R2: Add the calculated constant member to the start of the expression.
                e.values = [models.Primary(target=-adjusted_value)] + positive_factors

            # R5: Wrap the result in a negation.
            if len(e.values) == 1:
                # R6: Wrap by a negation and simplify the primary for future operations.
                return simplify_primary(models.Primary(sign="-", target=e.values[0]), recursion=False)
            else:
                return simplify_primary(models.Primary(sign="-", target=e), recursion=False)
        else:
            if adjusted_value == 1:
                # R4: Leave out the constant factor.
                e.values = positive_factors
            else:
                # R1, R2: Add the calculated constant member to the start of the expression.
                e.values = [models.Primary(target=adjusted_value)] + positive_factors
    else:
        # R3, R7: The default value of the adjusted value is 1.
        return models.Primary(target=adjusted_value)

    if len(e.values) == 1:
        # R6
        return e.values[0]
    else:
        return e


def merge_addition_terms(values: List[Union[models.Expression, models.Primary]]):
    """Attempt to find common objects in `values` and merge them mathematically if possible."""
    root_objects = dict()
    target_mapping = dict()

    # Group the values based on the primary component.
    for v in values:
        # Check if the members are negated primaries or not to determine the target to evaluate.
        factor = 1
        if isinstance(v, models.Primary) and v.sign != "":
            factor = -1
            target = v.body or v.ref
        else:
            target = v

        # Using the target, find factors that can be merged.
        if isinstance(target, models.Expression) and target.op == "*" and len(target.values) == 2:
            # By R5 and R8 of multiplication:
            #   - Multiplication factors are always positive and wrapped by a negation if appropriate.
            # By R2 of multiplication:
            #   - Constants are always at the start of the equation if present.
            if isinstance(target.values[0], models.Primary) and target.values[0].value is not None:
                factor *= target.values[0].signed_value
                target = target.values[1]

        v_str = str(target)
        target_mapping[v_str] = target
        root_objects[v_str] = root_objects.get(v_str, []) + [factor]

    # Reconstruct the values, based on the groups found.
    result = []
    for key, candidates in root_objects.items():
        target = target_mapping[key].create_copy({}, is_first=False)
        factor = sum(candidates)
        if factor != 0:
            result.append(simplify_multiplication(
                models.Expression("*", [models.Primary(target=factor), target])
            ))

    return result


# Addition
def simplify_addition(e: models.Expression):
    """
    Simplify the given addition, such that the following holds:
        - R1: Only one constant member remains after simplification.
        - R2: A constant member of zero is eliminated from the equation when appropriate.
        - R3: The constant member, when present, is the last member of the expression.
        - R4: A single value should be returned if the simplified operator is over a single value.
        - R5: An addition over no elements is equivalent to `0`.
        - R6: Complex members that are over the same object are consolidated into one (possibly empty) object.

    Assumptions:
        - A1: All members have been simplified before evaluation.

    Examples:
        - E1(R1, R3): `2 + expr1 + expr2 + 3` -> `expr1 + expr2 + 5`
        - E2(R1, R3, R6): `expr1 + 2 + 3 * expr2 + -(2 * expr2) + -expr2` -> `expr1 + 2`
        - E3(R1, R2, R4, R6): `expr1 + 0 + 3 * expr2 + -(2 * expr2) + -expr2` -> `expr1`
        - E4(R1, R2, R4): `1 + 2 + -3 -> `0`
        - E5: `expr1 + expr2` -> `expr1 + expr2`
        - E6: `expr1` -> `expr1`
        - E7(R5): `[]` -> `0`

    :param e: An expression object with the addition operator.
    :return: A simplification of input parameter e.
    """
    basic_values = [v for v in e.values if isinstance(v, models.Primary) and v.value is not None]
    adjusted_value = reduce(operator.__add__, [v.signed_value for v in basic_values], 0)
    values = [v for v in e.values if not isinstance(v, models.Primary) or v.value is None]

    # R6: Merge common terms if possible.
    values[:] = merge_addition_terms(values)

    # R2
    if adjusted_value != 0 or len(values) == 0:
        # R1, R3, R5: Falls back to an adjusted value of 0 when no values are included.
        values.append(models.Primary(target=adjusted_value))
    e.values = values

    if len(e.values) == 1:
        # R4
        return e.values[0]
    else:
        return e


def simplify_subtraction(e: models.Expression):
    """
    Simplify the given subtraction, such that the following holds:
        - R1: Only one constant member remains after simplification.
        - R2: A constant member of zero is eliminated from the equation when appropriate.
        - R3: The constant member, when present, is the last member of the expression.
        - R4: A single value should be returned if the simplified operator is over a single value.
        - R5: A subtraction over no elements is equivalent to `0`.
        - R6: Complex members that are over the same object are consolidated into one (possibly empty) object.
        - R7: The subtraction is converted to an addition to simplify the optimization process.

    Assumptions:
        - A1: All members have been simplified before evaluation.

    Examples:
        - E1(R1, R3, R7): `2 - expr1 - expr2 - 3` -> `-expr1 + -expr2 + -1`
        - E2(R1, R3, R6, R7): `expr1 - 2 - 3 * expr2 - -(2 * expr2) - -expr2` -> `expr1 + -2`
        - E3(R1, R2, R4, R6, R7): `expr1 - 0 - 3 * expr2 - -(2 * expr2) - -expr2` -> `expr1`
        - E4(R1, R2, R4, R7): `1 - -2 - 3 -> `0`
        - E5(R7): `expr1 - expr2` -> `expr1 + -(expr2)`
        - E6: `expr1` -> `expr1`
        - E7(R5): `[]` -> `0`

    :param e: An expression object with the subtraction operator.
    :return: A simplification of input parameter e, converted to an addition.
    """
    # R5
    if len(e.values) == 0:
        return models.Primary(target=0)

    # R7: Convert the subtraction to an addition of negated components.
    preprocessed_values = [e.values[0]]
    for v in e.values[1:]:
        if isinstance(v, models.Primary) and v.sign != "":
            if v.body is not None:
                # Note that the original body is wrapped by a negation.
                # Thus, taking out the body directly works, since only addition is the same priority and will be merged.
                preprocessed_values.append(v.body)
            else:
                preprocessed_values.append(models.Primary(target=v.ref or v.value))
        else:
            preprocessed_values.append(models.Primary("-", target=v))

    # R4
    if len(preprocessed_values) == 1:
        return preprocessed_values[0]

    # R1, R2, R3, R6: Convert to an addition and simplify the addition instead.
    # Simplify the expression to allow for the proper merging of the nested addition operators.
    return simplify_expression(models.Expression("+", preprocessed_values), recursion=False)


# Division
def simplify_division(e: models.Expression):
    """
    Simplify the given division, such that the following holds:
        - R1: A division by zero should be detected and propagated.
        - R2: A division by one becomes the dividend.
        - R3: A division of minus one becomes the negation of the dividend.
        - R4: A division of two equal values becomes one.
        - R5: A constant is returned when both members are a constant value.
        - R6: The division is wrapped in a negation if exactly one of the members is negative.
        - R7: The members within the division are never negative.
        - R8: A division with a zero dividend becomes zero.
        - TODO: R9: Common factors in divisions get cancelled out.
            (procedure needs to be implemented)

    Assumptions:
        - A1: All members have been simplified before evaluation.
        - A2: The modulo operator has exactly two members.

    Examples:
        - E1(R1): `expr1 / 0` -> `ZeroDivisionError()`
        - E2(A2): `expr1 / expr2 / expr3` -> `Exception()`
        - E3(R2): `expr1 / 1` -> `expr1`
        - E4(R3): `expr1 / -1` -> `-(expr1)`
        - E5(R4): `expr1 / expr1` -> `1`
        - E6(R4, R6, R7): `-expr1 / expr1` -> `-1`
        - E7(R5): `8 / 3` -> `2`
        - E8(R5): `8 / -3` -> `-3`
        - E9(R6, R7): `expr1 / -expr2` -> `-(expr1 / expr2)`
        - E10(R6, R7): `-expr1 / -expr2` -> `expr1 / expr2`
        - E11: `expr1 / expr2` -> `expr1 / expr2`
        - E12(R8): `expr1 * expr1 / expr1` -> `expr1`
        - (NYI) E12(R9): `expr1 * expr1 * expr2 / expr1 * expr2` -> `expr1`
        - (NYI) E13(R9): `expr1 ** 2 * expr2 * expr3 / expr1 * expr2` -> `expr1 * expr3`
        - (NYI) E14(R6, R7, R9): `-(expr1 * expr2 * expr3) / expr1 * expr3` -> `-(expr2)`
        - (NYI) E15(R9): `expr1 * expr2 / expr1 * expr3` -> `expr2 / expr3`
        - E16(R8): `0 / expr1` -> `0`

    :param e: An expression object with the division operator.
    :return: A simplification of input parameter e.
    """
    if len(e.values) != 2:
        # A2
        raise Exception("The division operator needs exactly two values.")

    if isinstance(e.values[1], models.Primary) and e.values[1].value == 0:
        # R1
        raise ZeroDivisionError("The expression", e, "has a division by zero.")
    elif isinstance(e.values[0], models.Primary) and e.values[0].value == 0:
        # R8
        return models.Primary(target=0)
    elif all(isinstance(v, models.Primary) and v.value is not None for v in e.values):
        # R5: Resolvable. Calculate the resulting value.
        # Note: Floor div doesn't exist in SMT. Integer division in SMT rounds up instead of down for negatives.
        # SMT solver result: https://rise4fun.com/Z3/NvKC
        value = int(operator.__truediv__(e.values[0].signed_value, e.values[1].signed_value))
        return models.Primary(target=value)
    else:
        # R7: Reduce the number of negations by only having one negation as a wrapper and using all positive factors.
        nr_of_negations = len([v for v in e.values if isinstance(v, models.Primary) and v.sign == "-"])
        positive_factors = []
        for v in e.values:
            if isinstance(v, models.Primary) and v.sign == "-":
                positive_factors.append(models.Primary(target=v.body or v.ref or v.value))
            else:
                positive_factors.append(v)

        # TODO: R8: Find common complex factors in the dividend and divisor and eliminate them from the equation.

        if positive_factors[0] == positive_factors[1]:
            # R4: A division of equal factors is equivalent to one.
            if nr_of_negations % 2 == 0:
                return models.Primary(target=1)
            else:
                # R6: One of the factors is negative, and hence the division is negative.
                return models.Primary(target=-1)
        elif isinstance(positive_factors[1], models.Primary) and positive_factors[1].value == 1:
            # R2: A division by one is equal to the dividend.
            if nr_of_negations % 2 == 0:
                return positive_factors[0]
            else:
                # R3, R6: One of the factors is negative, and hence the division is negative.
                return models.Primary(sign="-", target=positive_factors[0])
        else:
            e.values = positive_factors
            if nr_of_negations % 2 == 0:
                return e
            else:
                # R6: One of the factors is negative, and hence the division is negative.
                return models.Primary(sign="-", target=e)


# Modulo
def simplify_modulo(e: models.Expression):
    """
    Simplify the given modulo operation, such that the following holds:
        - R1: A division by zero should be detected and propagated.
        - R2: A modulo of one is simplified to zero.
        - R3: A modulo in which both members are equal is simplified to zero.
        - R4: A constant is returned when both members are a constant value.
        - R5: Common factors in modulo operators cancel out.
        - R6: The modulo of zero is always zero.

    Assumptions:
        - A1: All members have been simplified before evaluation.
        - A2: The modulo operator has exactly two members.
        - A3: Power operations within the modulo have been written out when possible for simplification.

    Examples:
        - E1(R1): `expr1 % 0` -> `ZeroDivisionError()`
        - E2(A2): `expr1 % expr2 % expr3` -> `Exception()`
        - E3(R2): `expr1 % 1` -> `0`
        - E4(R3): `expr1 % expr1` -> `0`
        - E5(R4): `8 % 3` -> `2`
        - E6(R4): `8 % -3` -> `-1`
        - E7: `expr1 % expr2` -> `expr1 % expr2`
        - E8(R5): `expr1 * expr2 % expr1` -> `0`
        - E9(R5): `expr1 * expr2 * expr3 % -expr1 * expr3` -> `0`
        - E10(R5): `expr1 * expr1 % -expr1` -> `0`
        - E11(R5): `4 * expr1 ** 2 % -(2 * expr1)` -> `0`
        - E12(R6): `0 % expr1` -> `0`


    :param e: An expression object with the modulo operator.
    :return: A simplification of input parameter e.
    """
    if len(e.values) != 2:
        # A2
        raise Exception("The modulo operator needs exactly two values.")

    if isinstance(e.values[1], models.Primary) and e.values[1].value == 0:
        # R1
        raise ZeroDivisionError("The expression", e, "has a modulo zero.")
    elif isinstance(e.values[1], models.Primary) and e.values[1].value == 1:
        # R2, R6: Modulo 1 is always 0 and the modulo of 0 is always 0.
        return models.Primary(target=0)
    elif isinstance(e.values[0], models.Primary) and e.values[0].value == 0:
        # R6: The modulo of 0 is always 0.
        return models.Primary(target=0)
    elif all(isinstance(v, models.Primary) and v.value is not None for v in e.values):
        # R4: Resolvable. Calculate the resulting value.
        value = operator.__mod__(e.values[0].signed_value, e.values[1].signed_value)
        return models.Primary(target=value)
    else:
        # Separately track the variables and the constant factors used within both parts of the statement.
        target_variables = [[], []]
        constant_factors = [1, 1]

        for i in range(0, 2):
            # Find the appropriate target, looking for compatible multiplication operations.
            target = e.values[i]

            # Readjust the target to the body of a primary if possible.
            if isinstance(e.values[i], models.Primary) and isinstance(e.values[i].body, models.Expression):
                target = e.values[i].body

            # Process the target with the appropriate multiplicity.
            if isinstance(target, models.Expression):
                if target.op == "*":
                    for v in target.values:
                        if isinstance(v, models.Primary):
                            if v.value is not None:
                                constant_factors[i] = v.value
                            else:
                                target_variables[i].append(v.body or v.ref)
                        else:
                            target_variables[i].append(v)
                else:
                    target_variables[i].append(target)
            elif target.value is not None:
                constant_factors[i] = target.value
            else:
                target_variables[i].append(target.ref)

        # R3, R5: Perform a subset check with multiplicity.
        if len(Counter(target_variables[1]) - Counter(target_variables[0])) == 0:
            # Ensure that the constant factors, if found, satisfy the same constraint.
            if constant_factors[1] == 1 or constant_factors[0] % constant_factors[1] == 0:
                return models.Primary(target=0)
    return e


# Power
def simplify_power(e: models.Expression):
    """
    Simplify the given power operation, such that the following holds:
        - R1: To the power of zero is simplified to one.
        - R2: The power of one is simplified to one.
        - R3: Negative exponents result in an exception.
        - R4: A constant is returned when both members are a constant value.
        - R5: Powers with constant exponents are converted to a multiplication.
        - TODO: R6: Addition in the exponent is broken up into two or more factors.
            (Procedure, examples and tests need to be implemented)

    Assumptions:
        - A1: All members have been simplified before evaluation.
        - A2: The power operator has exactly two members.

    Examples:
        - E1(A2): `expr1 ** expr2 ** expr3` -> `Exception()`
        - E2(R3): `expr1 ** -1` -> `Exception()`
        - E3(R1): `expr1 ** 0` -> `1`
        - E4(R2): `expr1 ** 1` -> `expr1`
        - E5(R4): `-2 ** 4` -> `16`
        - E6(R4): `-2 ** 3` -> `-8`
        - E7(R5): `expr1 ** 3` -> `expr1 * expr1 * expr1`
        - E8: `expr1 ** expr2` -> `expr1 ** expr2`

    :param e: An expression object with the power operator.
    :return: A simplification of input parameter e.
    """

    if len(e.values) != 2:
        # A2
        raise Exception("The power operator needs exactly two values.")

    if isinstance(e.values[1], models.Primary) and e.values[1].sign != "" and e.values[1].value is not None:
        # R3
        raise Exception("The power operator is incompatible with negative exponents.")

    if isinstance(e.values[1], models.Primary) and e.values[1].value == 0:
        # R1: To the power 0 is always 1.
        return models.Primary(target=1)
    elif isinstance(e.values[1], models.Primary) and e.values[1].value == 1:
        # R2: To the power 1 is always the value of the base.
        return e.values[0]
    elif all(isinstance(v, models.Primary) and v.value is not None for v in e.values):
        # R4: Resolvable. Calculate the resulting value.
        value = operator.__pow__(e.values[0].signed_value, e.values[1].signed_value)
        return models.Primary(target=value)
    elif isinstance(e.values[1], models.Primary) and e.values[1].value is not None:
        # R5: Write out the power as a multiplication and simplify accordingly.
        copies = [e.values[0]]
        for _ in range(1, e.values[1].value):
            copies.append(e.values[0].create_copy({}, is_first=False))
        return simplify_expression(models.Expression("*", copies), recursion=False)
    return e


# Logic operations
def simplify_comparison(e: models.Expression):
    """
    Simplify the given comparison operation, such that the following holds:
        - R1: Comparison operators with more than two members are converted to a conjunction of comparators.
        - R2: The comparison becomes `true` if the comparison always holds true.
        - R2: The comparison becomes `false` if the comparison never holds true.
        - R4: A constant is returned when both members are a constant value.

    Assumptions:
        - A1: All members have been simplified before evaluation.
        - A2: The comparator operator has at least two members.

    Examples:
        - E1(A2): `expr1` -> `Exception()`
        - E2(R1): `expr1 <op> expr2 <op> expr3` -> `expr1 <op> expr2 and expr2 <op> expr 3`
        - E3(R1, R2): `expr1 > expr1 > expr1 + 1` -> `false`
        - E4(R2): `expr1 < expr1 + 1` -> `true`
        - E5(R3): `expr1 = expr1 + 1` -> `false`
        - E6(R4): `-8 < 3` -> `true`
        - E7: `expr1 <op> expr2` -> `expr1 <op> expr2`

    :param e: An expression object with the comparison operator.
    :return: A simplification of input parameter e.
    """

    if len(e.values) < 2:
        # A2
        raise Exception("The comparison operator needs at least two values.")

    if len(e.values) > 2:
        # R1: Convert the combination of comparators to a conjunction of comparators.
        values = []
        for i in range(1, len(e.values)):
            # Make a recursive call to simplify further.
            values.append(simplify_comparison(models.Expression(e.op, [
                e.values[i - 1].create_copy({}, is_first=False),
                e.values[i].create_copy({}, is_first=False)
            ])))
        return simplify_expression(models.Expression("and", values), recursion=False)
    elif all(isinstance(v, models.Primary) and v.value is not None for v in e.values):
        # R4: Resolvable. Calculate the resulting value.
        return models.Primary(target=operator_mapping[e.op](e.values[0].signed_value, e.values[1].signed_value))
    elif e.is_true():
        # R2
        return models.Primary(target=True)
    elif e.is_false():
        # R3
        return models.Primary(target=False)
    else:
        return e


# A list of operators that are fully associative.
associative_operators = ["+", "*", "or", "and", "xor"]

# A list of operators that benefit from an n-ary approach.
mergeable_operators = associative_operators + ["-", "=", "!=", "<", ">", "<=", ">="]


def simplify_expression(e: models.Expression, recursion=True) -> Union[models.Expression, models.Primary]:
    """
    Simplify the given expression, such that the following holds:
        - R1: Operators that can be merged are merged with respect to the appropriate associativity rules.
        - R2: Negated additions nested within additions are merged by negating its members.
        - R3: Negated multiplications nested within multiplications are merged by negating the first member.
        - R4: All members have been simplified before evaluation, unless the recursion flag is `false`.

    Assumptions:
        - A2: The appropriate number of values are given: n > 2 for operators that can be merged, otherwise n = 2.

    Examples:
        - E1

    :param e: An expression object that needs to be simplified.
    :param recursion: A boolean denoting whether a recursive simplification call should be executed.
    :return: A simplification of input parameter e.
    """

    # R4
    if recursion:
        e.values = [simplify(v) for v in e.values]

    if len(e.values) >= 2 and e.op in mergeable_operators:
        # Merge the targets based on the associativity rules.
        values = []

        if e.op in associative_operators:
            # Associative operators don't have an order.
            target_values = e.values
        else:
            # Others are left-associative.
            # Note that the right-associativity of exponentiation isn't an issue, since it isn't a mergeable operator.
            target_values = e.values[:-1]

        # R1: Merge without violating the associativity rules.
        for v in target_values:
            if isinstance(v, models.Primary) and isinstance(v.body, models.Expression) and e.op == v.body.op:
                if v.sign == "":
                    values.extend(v.body.values)
                elif v.sign == "-" and e.op == "+":
                    # R2: Negate all members of the addition and merge.
                    values.extend([models.Primary(sign="-", target=n) for n in v.body.values])
                elif v.sign == "-" and e.op == "*":
                    # R3: Negate the first member of the multiplication and merge.
                    values.extend([models.Primary(sign="-", target=v.body.values[0])] + v.body.values[1:])
                else:
                    values.append(v)
            elif isinstance(v, models.Expression) and e.op == v.op:
                values.extend(v.values)
            else:
                values.append(v)

        # Add the last element back.
        if e.op not in associative_operators:
            values.append(e.values[-1])
        e.values = values

        return rewrite_functions[e.op](e)
    if len(e.values) == 2:
        return rewrite_functions[e.op](e)
    else:
        # A2
        raise Exception("The given expression has an unexpected number of values.")


# Negations of the given operators.
negation_table = {
    "=": "!=",
    "!=": "=",
    ">": "<=",
    "<": ">=",
    ">=": "<",
    "<=": ">",
}


def simplify_primary(e: models.Primary, recursion=True):
    """
    Simplify the given primary, such that the following holds:
        - R1: Nested negations are cancelled out.
        - R2: Operators that can be negated are negated properly.
        - R3: Constant values are simplified when possible.
        - R4: All members have been simplified before evaluation, unless the recursion flag is `false`.

    Examples:
        - E1(R1): `p1(body=expr1)` -> `expr1`
        - E2(R1): `p1(-,body=expr1)` -> `p1(-,body=expr1)`
        - E3(R2): `p1(-,p2(-,body=expr1))` -> `expr1`
        - E4(R1, R2): `p1(-,body=p2(body=p3(-,body=expr1))` -> `expr1`
        - E5(R3): `p1(-,body=expr1 <op> expr2)` -> `expr1 <negated_op> expr2`

    :param e: An primary object that needs to be simplified.
    :param recursion: A boolean denoting whether a recursive simplification call should be executed.
    :return: A simplification of input parameter e.
    """

    if e.body is not None:
        if recursion:
            # R4
            e.body = simplify(e.body)

        if isinstance(e.body, models.Primary):
            if e.body.value is not None:
                # R3: Elevate the basic value and apply the appropriate sign.
                if e.sign == "":
                    return e.body
                elif e.sign == "-":
                    return models.Primary(target=-e.body.signed_value)
                else:
                    return models.Primary(target=not e.body.signed_value)
            else:
                # TODO: Create tests for this situation.
                # Swap the signs.
                if e.sign in ["-", "not"] and e.sign == e.body.sign:
                    result = models.Primary(target=e.body.body or e.body.ref)
                else:
                    result = models.Primary(sign=e.sign or e.body.sign, target=e.body.body or e.body.ref)
                return simplify_primary(result, recursion=False)
        elif e.body.op in negation_table:
            # R2 Negate operators that can be negated.
            return models.Expression(negation_table[e.body.op], e.body.values)
    elif e.ref is not None:
        if recursion:
            # R4
            e.ref = simplify(e.ref)
    return e


def simplify_variable_ref(e: models.VariableRef):
    if e.index is not None:
        e.index = simplify(e.index)
    return e


def simplify_composite(e: models.Composite):
    e.guard = simplify(e.guard)
    e.assignments = [simplify(v) for v in e.assignments]
    return e


def simplify_assignment(e: models.Assignment):
    e.left = simplify(e.left)
    e.right = simplify(e.right)
    return e


def simplify_transition(e: models.Transition):
    e.statements = [simplify(s) for s in e.statements]
    return e


def simplify(e: Union[
    models.Expression, models.Primary, models.VariableRef, models.Composite, models.Assignment, models.Transition
]):
    """Simplify the given expression and preprocess calculations when possible."""
    if isinstance(e, models.Expression):
        e = simplify_expression(e)
    elif isinstance(e, models.Primary):
        e = simplify_primary(e)
    elif isinstance(e, models.VariableRef):
        e = simplify_variable_ref(e)
    elif isinstance(e, models.Composite):
        e = simplify_composite(e)
    elif isinstance(e, models.Assignment):
        e = simplify_assignment(e)
    elif isinstance(e, models.Transition):
        e = simplify_transition(e)
    return e


# Create a mapping to the rewrite functions.
rewrite_functions = {
    "and": simplify_conjunction,
    "or": simplify_disjunction,
    "xor": simplify_exclusive_disjunction,
    "*": simplify_multiplication,
    "+": simplify_addition,
    "-": simplify_subtraction,
    ">": simplify_comparison,
    "<": simplify_comparison,
    ">=": simplify_comparison,
    "<=": simplify_comparison,
    "=": simplify_comparison,
    "!=": simplify_comparison,
    "**": simplify_power,
    "/": simplify_division,
    "%": simplify_modulo,
}

# Map the operator string representation to the function.
operator_mapping = {
    ">": operator.__gt__,
    "<": operator.__lt__,
    ">=": operator.__ge__,
    "<=": operator.__le__,
    "=": operator.__eq__,
    "!=": operator.__ne__,
}
