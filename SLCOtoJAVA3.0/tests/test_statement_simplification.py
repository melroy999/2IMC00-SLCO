from unittest import TestCase, skip

from objects.ast.models import Expression, Primary, VariableRef, Variable, Type, Assignment, Composite, Transition
from preprocessing.statement_beautification import beautify
from preprocessing.statement_simplification import simplify_subtraction, simplify_addition, simplify_multiplication, \
    simplify_conjunction, simplify_disjunction, simplify_exclusive_disjunction, simplify_comparison, simplify_division, \
    simplify_modulo, simplify_power, simplify


# Get a dummy expression.
from util import smt


def math_expr_dummy(op="*", v_name="v", w_name="w"):
    v = Variable(v_name, _type=Type("Integer", 0))
    w = Variable(w_name, _type=Type("Integer", 0))
    return Expression(op, [Primary(target=VariableRef(var=v)), Primary(target=VariableRef(var=w))])


def logic_expr_dummy(op="and", v_name="a", w_name="b"):
    v = Variable(v_name, _type=Type("Boolean", 0))
    w = Variable(w_name, _type=Type("Boolean", 0))
    return Expression(op, [Primary(target=VariableRef(var=v)), Primary(target=VariableRef(var=w))])


def verify_simplification_assumption(test: TestCase, target):
    """Verify whether all members of the expression have been simplified."""
    for v in target:
        simplification = simplify(v.create_copy({}, is_first=False))
        test.assertEqual(v, simplification)
        test.assertEqual(str(v), str(simplification))


def verify_structure_integrity(test: TestCase, node, parent=None):
    """Verify whether the parent-child structure is kept intact."""
    if parent is not None:
        test.assertIs(node.parent, parent)
        test.assertEqual(str(node.parent), str(parent))
    for v in node:
        verify_structure_integrity(test, v, node)


# noinspection DuplicatedCode
class TestSimplifyConjunction(TestCase):
    def setUp(self) -> None:
        smt.clear_smt_cache()

    def test_e1(self):
        """E1(R1): `a and b and true` -> `a and b`"""
        a = Variable("a", _type=Type("Boolean", 0))
        b = Variable("b", _type=Type("Boolean", 0))

        e = Expression("and", [
            Primary(target=VariableRef(a)),
            Primary(target=VariableRef(b)),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "a and b and true")
        verify_simplification_assumption(self, e)
        result = simplify_conjunction(e)

        self.assertEqual(result, Expression("and", [
            Primary(target=VariableRef(a)),
            Primary(target=VariableRef(b))
        ]))
        self.assertEqual(str(result), "a and b")
        verify_structure_integrity(self, result)

    def test_e2(self):
        """E2(R3): `x > 0 and x > 0 and y > 0` -> `x > 0 and y > 0`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))

        e = Expression("and", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(y)), Primary(target=0)]),
        ])

        self.assertEqual(str(e), "x > 0 and x > 0 and y > 0")
        verify_simplification_assumption(self, e)
        result = simplify_conjunction(e)

        self.assertEqual(result, Expression("and", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(y)), Primary(target=0)])
        ]))
        self.assertEqual(str(result), "x > 0 and y > 0")
        verify_structure_integrity(self, result)

    def test_e3(self):
        """E3(R2): `!a and !b and false` -> `false`"""
        a = Variable("a", _type=Type("Boolean", 0))
        b = Variable("b", _type=Type("Boolean", 0))

        e = Expression("and", [
            Primary(sign="not", target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(b)),
            Primary(target=False)
        ])

        self.assertEqual(str(e), "!a and !b and false")
        verify_simplification_assumption(self, e)
        result = simplify_conjunction(e)

        self.assertEqual(result, Primary(target=False))
        self.assertEqual(str(result), "false")
        verify_structure_integrity(self, result)

    def test_e4(self):
        """E4(R6): `x > 0 and x <= 0` -> `false`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("and", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression("<=", [Primary(target=VariableRef(x)), Primary(target=0)]),
        ])

        self.assertEqual(str(e), "x > 0 and x <= 0")
        verify_simplification_assumption(self, e)
        result = simplify_conjunction(e)

        self.assertEqual(result, Primary(target=False))
        self.assertEqual(str(result), "false")
        verify_structure_integrity(self, result)

    def test_e5(self):
        """E5(R1, R4): `true and true` -> `true`"""
        e = Expression("and", [
            Primary(target=True),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "true and true")
        verify_simplification_assumption(self, e)
        result = simplify_conjunction(e)

        self.assertEqual(result, Primary(target=True))
        self.assertEqual(str(result), "true")
        verify_structure_integrity(self, result)

    def test_e6(self):
        """E6(R1, R4): `a and true` -> `true`"""
        a = Variable("a", _type=Type("Boolean", 0))

        e = Expression("and", [
            Primary(target=VariableRef(a)),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "a and true")
        verify_simplification_assumption(self, e)
        result = simplify_conjunction(e)

        self.assertEqual(result, Primary(target=VariableRef(a)))
        self.assertEqual(str(result), "a")
        verify_structure_integrity(self, result)

    def test_e7(self):
        """E7: `!a and !b` -> `!a and !b`"""
        a = Variable("a", _type=Type("Boolean", 0))
        b = Variable("b", _type=Type("Boolean", 0))

        e = Expression("and", [
            Primary(sign="not", target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(b)),
        ])

        self.assertEqual(str(e), "!a and !b")
        verify_simplification_assumption(self, e)
        result = simplify_conjunction(e)

        self.assertEqual(result, Expression("and", [
            Primary(sign="not", target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(b)),
        ]))
        self.assertEqual(str(result), "!a and !b")
        verify_structure_integrity(self, result)

    def test_e8(self):
        """E8: `x > 0` -> `x > 0`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("and", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
        ])

        self.assertEqual(str(e), "x > 0")
        verify_simplification_assumption(self, e)
        result = simplify_conjunction(e)

        self.assertEqual(result, Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]))
        self.assertEqual(str(result), "x > 0")
        verify_structure_integrity(self, result)

    def test_e9(self):
        """E9: `[]` -> `true`"""
        e = Expression("and", [])

        self.assertEqual(str(e), "")
        verify_simplification_assumption(self, e)
        result = simplify_conjunction(e)

        self.assertEqual(result, Primary(target=True))
        self.assertEqual(str(result), "true")
        verify_structure_integrity(self, result)


class TestSimplifyDisjunction(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_e1(self):
        """E1(R1): `a or b or false` -> `a or b`"""
        a = Variable("a", _type=Type("Boolean", 0))
        b = Variable("b", _type=Type("Boolean", 0))

        e = Expression("or", [
            Primary(target=VariableRef(a)),
            Primary(target=VariableRef(b)),
            Primary(target=False)
        ])

        self.assertEqual(str(e), "a or b or false")
        verify_simplification_assumption(self, e)
        result = simplify_disjunction(e)

        self.assertEqual(result, Expression("or", [
            Primary(target=VariableRef(a)),
            Primary(target=VariableRef(b))
        ]))
        self.assertEqual(str(result), "a or b")
        verify_structure_integrity(self, result)

    def test_e2(self):
        """E2(R3): `x > 0 or x > 0 or y > 0` -> `x > 0 or y > 0`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))

        e = Expression("or", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(y)), Primary(target=0)]),
        ])

        self.assertEqual(str(e), "x > 0 or x > 0 or y > 0")
        verify_simplification_assumption(self, e)
        result = simplify_disjunction(e)

        self.assertEqual(result, Expression("or", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(y)), Primary(target=0)])
        ]))
        self.assertEqual(str(result), "x > 0 or y > 0")
        verify_structure_integrity(self, result)

    def test_e3(self):
        """E3(R2): `!a or !b or true` -> `true`"""
        a = Variable("a", _type=Type("Boolean", 0))
        b = Variable("b", _type=Type("Boolean", 0))

        e = Expression("or", [
            Primary(sign="not", target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(b)),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "!a or !b or true")
        verify_simplification_assumption(self, e)
        result = simplify_disjunction(e)

        self.assertEqual(result, Primary(target=True))
        self.assertEqual(str(result), "true")
        verify_structure_integrity(self, result)

    def test_e4(self):
        """E4(R6): `x > 0 or x <= 0` -> `true`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("or", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression("<=", [Primary(target=VariableRef(x)), Primary(target=0)]),
        ])

        self.assertEqual(str(e), "x > 0 or x <= 0")
        verify_simplification_assumption(self, e)
        result = simplify_disjunction(e)

        self.assertEqual(result, Primary(target=True))
        self.assertEqual(str(result), "true")
        verify_structure_integrity(self, result)

    def test_e5(self):
        """E5(R1, R4): `false or false` -> `false`"""
        e = Expression("or", [
            Primary(target=False),
            Primary(target=False)
        ])

        self.assertEqual(str(e), "false or false")
        verify_simplification_assumption(self, e)
        result = simplify_disjunction(e)

        self.assertEqual(result, Primary(target=False))
        self.assertEqual(str(result), "false")
        verify_structure_integrity(self, result)

    def test_e6(self):
        """E6(R1, R4): `a or false` -> `a`"""
        a = Variable("a", _type=Type("Boolean", 0))

        e = Expression("or", [
            Primary(target=VariableRef(a)),
            Primary(target=False)
        ])

        self.assertEqual(str(e), "a or false")
        verify_simplification_assumption(self, e)
        result = simplify_disjunction(e)

        self.assertEqual(result, Primary(target=VariableRef(a)))
        self.assertEqual(str(result), "a")
        verify_structure_integrity(self, result)

    def test_e7(self):
        """E7: `!a or !b` -> `!a or !b`"""
        a = Variable("a", _type=Type("Boolean", 0))
        b = Variable("b", _type=Type("Boolean", 0))

        e = Expression("or", [
            Primary(sign="not", target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(b)),
        ])

        self.assertEqual(str(e), "!a or !b")
        verify_simplification_assumption(self, e)
        result = simplify_disjunction(e)

        self.assertEqual(result, Expression("or", [
            Primary(sign="not", target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(b)),
        ]))
        self.assertEqual(str(result), "!a or !b")
        verify_structure_integrity(self, result)

    def test_e8(self):
        """E8: `x > 0` -> `x > 0`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("or", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
        ])

        self.assertEqual(str(e), "x > 0")
        verify_simplification_assumption(self, e)
        result = simplify_disjunction(e)

        self.assertEqual(result, Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]))
        self.assertEqual(str(result), "x > 0")
        verify_structure_integrity(self, result)

    def test_e9(self):
        """E9: `[]` -> `false`"""
        e = Expression("or", [])

        self.assertEqual(str(e), "")
        verify_simplification_assumption(self, e)
        result = simplify_disjunction(e)

        self.assertEqual(result, Primary(target=False))
        self.assertEqual(str(result), "false")
        verify_structure_integrity(self, result)


class TestSimplifyExclusiveDisjunction(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_e1(self):
        """E1(R6): `a xor b xor false xor false` -> `a xor b`"""
        a = Variable("a", _type=Type("Boolean", 0))
        b = Variable("b", _type=Type("Boolean", 0))

        e = Expression("xor", [
            Primary(target=VariableRef(a)),
            Primary(target=VariableRef(b)),
            Primary(target=False),
            Primary(target=False)
        ])

        self.assertEqual(str(e), "a xor b xor false xor false")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Expression("xor", [
            Primary(target=VariableRef(a)),
            Primary(target=VariableRef(b)),
        ]))
        self.assertEqual(str(result), "a xor b")
        verify_structure_integrity(self, result)

    def test_e2(self):
        """E2(R2): `a xor true xor b xor false` -> `a xor !b`"""
        a = Variable("a", _type=Type("Boolean", 0))
        b = Variable("b", _type=Type("Boolean", 0))

        e = Expression("xor", [
            Primary(target=VariableRef(a)),
            Primary(target=True),
            Primary(target=VariableRef(b)),
            Primary(target=False)
        ])

        self.assertEqual(str(e), "a xor true xor b xor false")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Expression("xor", [
            Primary(target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(b)),
        ]))
        self.assertEqual(str(result), "a xor !b")
        verify_structure_integrity(self, result)

    def test_e3(self):
        """E3(R2, R3): `x > 0 xor x > 0 xor x > 0 xor true` -> `x <= 0`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("xor", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "x > 0 xor x > 0 xor x > 0 xor true")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Expression("<=", [Primary(target=VariableRef(x)), Primary(target=0)]))
        self.assertEqual(str(result), "x <= 0")
        verify_structure_integrity(self, result)

    def test_e4_1(self):
        """E4(R4, R5): `a xor !a xor a xor true` -> `a`"""
        a = Variable("a", _type=Type("Boolean", 0))

        e = Expression("xor", [
            Primary(target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(a)),
            Primary(target=VariableRef(a)),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "a xor !a xor a xor true")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Primary(target=VariableRef(a)))
        self.assertEqual(str(result), "a")
        verify_structure_integrity(self, result)

    def test_e4_2(self):
        """E4(R4, R5): `x > 0 xor x <= 0 xor x > 0 xor true` -> `x > 0`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("xor", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression("<=", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "x > 0 xor x <= 0 xor x > 0 xor true")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]))
        self.assertEqual(str(result), "x > 0")
        verify_structure_integrity(self, result)

    def test_e5_1(self):
        """E5(R4, R5, R6): `!a xor a xor !a xor true` -> `!a`"""
        a = Variable("a", _type=Type("Boolean", 0))

        e = Expression("xor", [
            Primary(sign="not", target=VariableRef(a)),
            Primary(target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(a)),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "!a xor a xor !a xor true")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Primary(sign="not", target=VariableRef(a)))
        self.assertEqual(str(result), "!a")
        verify_structure_integrity(self, result)

    def test_e5_2(self):
        """E5(R4, R5, R6): `x <= 0 xor x > 0 xor x > 0 xor true` -> `x <= 0`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("xor", [
            Expression("<=", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "x <= 0 xor x > 0 xor x > 0 xor true")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Expression(">", [
            Primary(target=VariableRef(x)),
            Primary(target=0)
        ]))
        self.assertEqual(str(result), "x > 0")
        verify_structure_integrity(self, result)

    def test_e6_1(self):
        """E6(R3, R5): `!a xor !a xor true` -> `true`"""
        a = Variable("a", _type=Type("Boolean", 0))

        e = Expression("xor", [
            Primary(sign="not", target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(a)),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "!a xor !a xor true")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Primary(target=True))
        self.assertEqual(str(result), "true")
        verify_structure_integrity(self, result)

    def test_e6_2(self):
        """E6(R3, R5): `x <= 0 xor x <= 0 xor true` -> `true`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("xor", [
            Expression("<=", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression("<=", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "x <= 0 xor x <= 0 xor true")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Primary(target=True))
        self.assertEqual(str(result), "true")
        verify_structure_integrity(self, result)

    def test_e7_1(self):
        """E7(R4, R5): `a xor !a xor true` -> `false`"""
        a = Variable("a", _type=Type("Boolean", 0))

        e = Expression("xor", [
            Primary(target=VariableRef(a)),
            Primary(sign="not", target=VariableRef(a)),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "a xor !a xor true")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Primary(target=False))
        self.assertEqual(str(result), "false")
        verify_structure_integrity(self, result)

    def test_e7_2(self):
        """E7(R4, R5): `x > 0 xor x <= 0 xor true` -> `false`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("xor", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression("<=", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Primary(target=True)
        ])

        self.assertEqual(str(e), "x > 0 xor x <= 0 xor true")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Primary(target=False))
        self.assertEqual(str(result), "false")
        verify_structure_integrity(self, result)

    def test_e8(self):
        """E8: `x > 0 xor y > 0` -> `x > 0 xor y > 0`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))

        e = Expression("xor", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(y)), Primary(target=0)])
        ])

        self.assertEqual(str(e), "x > 0 xor y > 0")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Expression("xor", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(y)), Primary(target=0)])
        ]))
        self.assertEqual(str(result), "x > 0 xor y > 0")
        verify_structure_integrity(self, result)

    def test_e9(self):
        """E9: `x > 0` -> `x > 0`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("xor", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
        ])

        self.assertEqual(str(e), "x > 0")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]))
        self.assertEqual(str(result), "x > 0")
        verify_structure_integrity(self, result)

    def test_e10(self):
        """E10(R6): `[]` -> `false`"""
        e = Expression("xor", [])

        self.assertEqual(str(e), "")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Primary(target=False))
        self.assertEqual(str(result), "false")
        verify_structure_integrity(self, result)

    def test_e11(self):
        """E11(R3, R5): `x > 0 xor x > 0` -> `false`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("xor", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),

        ])

        self.assertEqual(str(e), "x > 0 xor x > 0")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Primary(target=False))
        self.assertEqual(str(result), "false")
        verify_structure_integrity(self, result)

    def test_e12(self):
        """E12(R4, R5): `x > 0 xor x <= 0` -> `true`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("xor", [
            Expression(">", [Primary(target=VariableRef(x)), Primary(target=0)]),
            Expression("<=", [Primary(target=VariableRef(x)), Primary(target=0)]),
        ])

        self.assertEqual(str(e), "x > 0 xor x <= 0")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Primary(target=True))
        self.assertEqual(str(result), "true")
        verify_structure_integrity(self, result)

    def test_e13(self):
        """E13(R1): `true xor false` -> `true`"""
        e = Expression("xor", [
            Primary(target=True),
            Primary(target=False),
        ])

        self.assertEqual(str(e), "true xor false")
        verify_simplification_assumption(self, e)
        result = simplify_exclusive_disjunction(e)

        self.assertEqual(result, Primary(target=True))
        self.assertEqual(str(result), "true")
        verify_structure_integrity(self, result)


class TestSimplifyMultiplication(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_e1(self):
        """E1(R5, R8): `-2 * (x + 1) * x` -> `-(2 * (x + 1) * x)`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("*", [
            Primary(target=-2),
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(target=VariableRef(x)),
        ])

        self.assertEqual(str(e), "-2 * (x + 1) * x")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Primary(sign="-", target=Expression("*", [
            Primary(target=2),
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(target=VariableRef(x)),
        ])))
        self.assertEqual(str(result), "-(2 * (x + 1) * x)")
        verify_structure_integrity(self, result)

    def test_e2(self):
        """E2(R4, R5): `-1 * (x + 1) * x` -> `-((x + 1) * x)`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("*", [
            Primary(target=-1),
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(target=VariableRef(x)),
        ])

        self.assertEqual(str(e), "-1 * (x + 1) * x")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Primary(sign="-", target=Expression("*", [
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(target=VariableRef(x)),
        ])))
        self.assertEqual(str(result), "-((x + 1) * x)")
        verify_structure_integrity(self, result)

    def test_e3(self):
        """E3(R3): `0 * x + 1 * x` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("*", [
            Primary(target=0),
            Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)]),
            Primary(target=VariableRef(x)),
        ])

        self.assertEqual(str(e), "0 * x + 1 * x")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e4(self):
        """E4(R4): `1 * (x + 1) * x` -> `(x + 1) * x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("*", [
            Primary(target=1),
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(target=VariableRef(x)),
        ])

        self.assertEqual(str(e), "1 * (x + 1) * x")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Expression("*", [
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(target=VariableRef(x)),
        ]))
        self.assertEqual(str(result), "(x + 1) * x")
        verify_structure_integrity(self, result)

    def test_e5(self):
        """E5(R1, R2): `2 * (x + 1) * x * 4` -> `8 * (x + 1) * x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("*", [
            Primary(target=2),
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(target=VariableRef(x)),
            Primary(target=4)
        ])

        self.assertEqual(str(e), "2 * (x + 1) * x * 4")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Expression("*", [
            Primary(target=8),
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(target=VariableRef(x)),
        ]))
        self.assertEqual(str(result), "8 * (x + 1) * x")
        verify_structure_integrity(self, result)

    def test_e6(self):
        """E6(R5, R8): `(x + 1) * -x` -> `-((x + 1) * x)`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("*", [
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(sign="-", target=VariableRef(x))
        ])

        self.assertEqual(str(e), "(x + 1) * -x")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Primary(sign="-", target=Expression("*", [
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(target=VariableRef(x))
        ])))
        self.assertEqual(str(result), "-((x + 1) * x)")
        verify_structure_integrity(self, result)

    def test_e7(self):
        """E7(R2, R8): `(x + 1) * -x * -2` -> `2 * (x + 1) * x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("*", [
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(sign="-", target=VariableRef(x)),
            Primary(target=-2)
        ])

        self.assertEqual(str(e), "(x + 1) * -x * -2")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Expression("*", [
            Primary(target=2),
            Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
            Primary(target=VariableRef(x))
        ]))
        self.assertEqual(str(result), "2 * (x + 1) * x")
        verify_structure_integrity(self, result)

    def test_e8(self):
        """E8(R4): `1 * (x + 1)` -> `x + 1`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("*", [Primary(target=1), Primary(target=Expression("+", [
            Primary(target=VariableRef(x)),
            Primary(target=1)
        ]))])

        self.assertEqual(str(e), "1 * (x + 1)")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)]))
        self.assertEqual(str(result), "(x + 1)")
        verify_structure_integrity(self, result)

    def test_e9(self):
        """E9(R4, R6): `-1 * (x + 1)` -> `-(x + 1)`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("*", [Primary(target=-1), Primary(target=Expression("+", [
            Primary(target=VariableRef(x)),
            Primary(target=1)
        ]))])

        self.assertEqual(str(e), "-1 * (x + 1)")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Primary(sign="-", target=Expression("+", [
            Primary(target=VariableRef(x)),
            Primary(target=1)
        ])))
        self.assertEqual(str(result), "-(x + 1)")
        verify_structure_integrity(self, result)

    def test_e10(self):
        """E10(R1, R2, R6): `-1 * 2` -> `-2`"""
        e = Expression("*", [Primary(target=-1), Primary(target=2)])

        self.assertEqual(str(e), "-1 * 2")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Primary(target=-2))
        self.assertEqual(str(result), "-2")
        verify_structure_integrity(self, result)

    def test_e11(self):
        """E11: `x * y` -> `x * y`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("*", [Primary(target=VariableRef(x)), Primary(target=VariableRef(y))])

        self.assertEqual(str(e), "x * y")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Expression("*", [Primary(target=VariableRef(x)), Primary(target=VariableRef(y))]))
        self.assertEqual(str(result), "x * y")
        verify_structure_integrity(self, result)

    def test_e12(self):
        """E12: `x` -> `x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("*", [Primary(target=VariableRef(x))])

        self.assertEqual(str(e), "x")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Primary(target=VariableRef(x)))
        self.assertEqual(str(result), "x")
        verify_structure_integrity(self, result)

    def test_e13(self):
        """E13(R7): `[]` -> `1`"""
        e = Expression("*", [])

        self.assertEqual(str(e), "")
        verify_simplification_assumption(self, e)
        result = simplify_multiplication(e)

        self.assertEqual(result, Primary(target=1))
        self.assertEqual(str(result), "1")
        verify_structure_integrity(self, result)


class TestSimplifyAddition(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_e1(self):
        """E1(R1, R3): `2 + 2 * x + y * x + 3 / x + 3` -> `2 * x + y * x + 3 / x + 5`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("+", [
            Primary(target=2),
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]),
            Expression("*", [Primary(target=VariableRef(y)), Primary(target=VariableRef(x))]),
            Expression("/", [Primary(target=3), Primary(target=VariableRef(x))]),
            Primary(target=3)
        ])

        self.assertEqual(str(e), "2 + 2 * x + y * x + 3 / x + 3")
        verify_simplification_assumption(self, e)
        result = simplify_addition(e)

        self.assertEqual(result, Expression("+", [
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]),
            Expression("*", [Primary(target=VariableRef(y)), Primary(target=VariableRef(x))]),
            Expression("/", [Primary(target=3), Primary(target=VariableRef(x))]),
            Primary(target=5)
        ]))
        self.assertEqual(str(result), "2 * x + y * x + 3 / x + 5")
        verify_structure_integrity(self, result)

    def test_e2(self):
        """E2(R1, R3, R6): `x * x + 2 + 3 * x + -(2 * x) + -x` -> `x * x + 2`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("+", [
            Expression("*", [Primary(target=VariableRef(x))] * 2),
            Primary(target=2),
            Expression("*", [Primary(target=3), Primary(target=VariableRef(x))]),
            Primary(sign="-", target=Expression("*", [Primary(target=2), Primary(target=VariableRef(x))])),
            Primary(sign="-", target=VariableRef(x))
        ])

        self.assertEqual(str(e), "x * x + 2 + 3 * x + -(2 * x) + -x")
        verify_simplification_assumption(self, e)
        result = simplify_addition(e)

        self.assertEqual(result, Expression("+", [
            Expression("*", [Primary(target=VariableRef(x))] * 2),
            Primary(target=2)
        ]))
        self.assertEqual(str(result), "x * x + 2")
        verify_structure_integrity(self, result)

    def test_e3(self):
        """E3(R1, R2, R4, R6): `x * x + 0 + 3 * x + -(2 * x) + -x` -> `x * x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("+", [
            Expression("*", [Primary(target=VariableRef(x))] * 2),
            Primary(target=0),
            Expression("*", [Primary(target=3), Primary(target=VariableRef(x))]),
            Primary(sign="-", target=Expression("*", [Primary(target=2), Primary(target=VariableRef(x))])),
            Primary(sign="-", target=VariableRef(x))
        ])

        self.assertEqual(str(e), "x * x + 0 + 3 * x + -(2 * x) + -x")
        verify_simplification_assumption(self, e)
        result = simplify_addition(e)

        self.assertEqual(result, Expression("*", [Primary(target=VariableRef(x))] * 2))
        self.assertEqual(str(result), "x * x")
        verify_structure_integrity(self, result)

    def test_e4(self):
        """E4(R1, R2, R4): `1 + 2 + -3 -> `0`"""
        e = Expression("+", [
            Primary(target=1),
            Primary(target=2),
            Primary(target=-3)
        ])

        self.assertEqual(str(e), "1 + 2 + -3")
        verify_simplification_assumption(self, e)
        result = simplify_addition(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e5(self):
        """E5: `2 * x + 3 / x` -> `2 * x + 3 / x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("+", [
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]),
            Expression("/", [Primary(target=3), Primary(target=VariableRef(x))]),
        ])

        self.assertEqual(str(e), "2 * x + 3 / x")
        verify_simplification_assumption(self, e)
        result = simplify_addition(e)

        self.assertEqual(result, Expression("+", [
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]),
            Expression("/", [Primary(target=3), Primary(target=VariableRef(x))]),
        ]))
        self.assertEqual(str(result), "2 * x + 3 / x")
        verify_structure_integrity(self, result)

    def test_e6(self):
        """E6: `2 * x` -> `2 * x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("+", [Expression("*", [Primary(target=2), Primary(target=VariableRef(x))])])

        self.assertEqual(str(e), "2 * x")
        verify_simplification_assumption(self, e)
        result = simplify_addition(e)

        self.assertEqual(result, Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]))
        self.assertEqual(str(result), "2 * x")
        verify_structure_integrity(self, result)

    def test_e7(self):
        """E7(R5): `[]` -> `0`"""
        e = Expression("+", [])

        self.assertEqual(str(e), "")
        verify_simplification_assumption(self, e)
        result = simplify_addition(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)


class TestSimplifySubtraction(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_e1(self):
        """E1(R1, R3, R7): `1 - 2 * x - y * x - 3 / x - 3` -> `-(2 * x) + -(y * x) + -(3 / x) + -1`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("-", [
            Primary(target=1),
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]),
            Expression("*", [Primary(target=VariableRef(y)), Primary(target=VariableRef(x))]),
            Expression("/", [Primary(target=3), Primary(target=VariableRef(x))]),
            Primary(target=3)
        ])

        self.assertEqual(str(e), "1 - 2 * x - y * x - 3 / x - 3")
        verify_simplification_assumption(self, e)
        result = simplify_subtraction(e)

        self.assertEqual(result, Expression("+", [
            Primary(sign="-", target=Expression("*", [Primary(target=2), Primary(target=VariableRef(x))])),
            Primary(sign="-", target=Expression("*", [Primary(target=VariableRef(y)), Primary(target=VariableRef(x))])),
            Primary(sign="-", target=Expression("/", [Primary(target=3), Primary(target=VariableRef(x))])),
            Primary(target=-2)
        ]))
        self.assertEqual(str(result), "-(2 * x) + -(y * x) + -(3 / x) + -2")
        verify_structure_integrity(self, result)

    def test_e2(self):
        """E2(R1, R3, R6, R7): `x * x - 3 - 3 * x - -(2 * x) - -x` -> `x ** 2 + -3`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("-", [
            Expression("*", [Primary(target=VariableRef(x))] * 2),
            Primary(target=3),
            Expression("*", [Primary(target=3), Primary(target=VariableRef(x))]),
            Primary(sign="-", target=Expression("*", [Primary(target=2), Primary(target=VariableRef(x))])),
            Primary(sign="-", target=VariableRef(x))
        ])

        self.assertEqual(str(e), "x * x - 3 - 3 * x - -(2 * x) - -x")
        verify_simplification_assumption(self, e)
        result = simplify_subtraction(e)

        self.assertEqual(result, Expression("+", [
            Expression("*", [Primary(target=VariableRef(x))] * 2),
            Primary(target=-3)
        ]))
        self.assertEqual(str(result), "x * x + -3")
        verify_structure_integrity(self, result)

    def test_e3(self):
        """E3(R1, R2, R4, R6, R7): `x * x * x - 0 - 3 * x - -(2 * x) - -x` -> `x * x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("-", [
            Expression("*", [Primary(target=VariableRef(x))] * 3),
            Primary(target=0),
            Expression("*", [Primary(target=3), Primary(target=VariableRef(x))]),
            Primary(sign="-", target=Expression("*", [Primary(target=2), Primary(target=VariableRef(x))])),
            Primary(sign="-", target=VariableRef(x))
        ])

        self.assertEqual(str(e), "x * x * x - 0 - 3 * x - -(2 * x) - -x")
        verify_simplification_assumption(self, e)
        result = simplify_subtraction(e)

        self.assertEqual(result, Expression("*", [Primary(target=VariableRef(x))] * 3))
        self.assertEqual(str(result), "x * x * x")
        verify_structure_integrity(self, result)

    def test_e4(self):
        """E4(R1, R2, R4, R7): `1 - -2 - 3 -> `0`"""
        e = Expression("-", [
            Primary(target=1),
            Primary(target=-2),
            Primary(target=3)
        ])

        self.assertEqual(str(e), "1 - -2 - 3")
        verify_simplification_assumption(self, e)
        result = simplify_subtraction(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e5(self):
        """E5: `2 * x - 3 / x` -> `2 * x + -(3 / x)`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("-", [
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]),
            Expression("/", [Primary(target=3), Primary(target=VariableRef(x))]),
        ])

        self.assertEqual(str(e), "2 * x - 3 / x")
        verify_simplification_assumption(self, e)
        result = simplify_subtraction(e)

        self.assertEqual(result, Expression("+", [
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]),
            Primary(sign="-", target=Expression("/", [Primary(target=3), Primary(target=VariableRef(x))])),
        ]))
        self.assertEqual(str(result), "2 * x + -(3 / x)")
        verify_structure_integrity(self, result)

    def test_e6(self):
        """E6: `2 * x` -> `2 * x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("-", [Expression("*", [Primary(target=2), Primary(target=VariableRef(x))])])

        self.assertEqual(str(e), "2 * x")
        verify_simplification_assumption(self, e)
        result = simplify_subtraction(e)

        self.assertEqual(result, Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]))
        self.assertEqual(str(result), "2 * x")
        verify_structure_integrity(self, result)

    def test_e7(self):
        """E7(R5): `[]` -> `0`"""
        e = Expression("-", [])

        self.assertEqual(str(e), "")
        verify_simplification_assumption(self, e)
        result = simplify_subtraction(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)


class TestSimplifyDivision(TestCase):
    # TODO: has tests with power operations that cannot occur during simplification.
    def setUp(self):
        smt.clear_smt_cache()

    def test_e1(self):
        """E1(R1): `x / 0` -> `ZeroDivisionError()`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("/", [Primary(target=VariableRef(x)), Primary(target=0)])

        self.assertEqual(str(e), "x / 0")
        self.assertRaises(ZeroDivisionError, simplify_division, e)

    def test_e2(self):
        """E2(A2): `x / x / x` -> `Exception()`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("/", [Primary(target=VariableRef(x))] * 3)

        self.assertEqual(str(e), "x / x / x")
        self.assertRaises(Exception, simplify_division, e)

    def test_e3(self):
        """E3(R2): `x * x / 1` -> `x ** 2`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("/", [Expression("*", [Primary(target=VariableRef(x))] * 2), Primary(target=1)])

        self.assertEqual(str(e), "x * x / 1")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Expression("*", [Primary(target=VariableRef(x))] * 2))
        self.assertEqual(str(result), "x * x")
        verify_structure_integrity(self, result)

    def test_e4(self):
        """E4(R3): `x / -1` -> `-x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("/", [Primary(target=VariableRef(x)), Primary(target=-1)])

        self.assertEqual(str(e), "x / -1")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Primary(sign="-", target=VariableRef(x)))
        self.assertEqual(str(result), "-x")
        verify_structure_integrity(self, result)

    def test_e5(self):
        """E5(R4): `x * y / (y * x)` -> `1`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("/", [
            Expression("*", [Primary(target=VariableRef(x)), Primary(target=VariableRef(y))]),
            Primary(target=Expression("*", [Primary(target=VariableRef(y)), Primary(target=VariableRef(x))])),
        ])

        self.assertEqual(str(e), "x * y / (y * x)")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Primary(target=1))
        self.assertEqual(str(result), "1")
        verify_structure_integrity(self, result)

    def test_e6(self):
        """E6(R4, R6, R7): `-(x * y) / (y * x)` -> `1`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("/", [
            Primary(sign="-", target=Expression("*", [Primary(target=VariableRef(x)), Primary(target=VariableRef(y))])),
            Primary(target=Expression("*", [Primary(target=VariableRef(y)), Primary(target=VariableRef(x))])),
        ])

        self.assertEqual(str(e), "-(x * y) / (y * x)")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Primary(target=-1))
        self.assertEqual(str(result), "-1")
        verify_structure_integrity(self, result)

    def test_e7(self):
        """E7(R5): `8 / 3` -> `2`"""
        e = Expression("/", [Primary(target=8), Primary(target=3)])

        self.assertEqual(str(e), "8 / 3")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Primary(target=2))
        self.assertEqual(str(result), "2")
        verify_structure_integrity(self, result)

    def test_e8(self):
        """E8(R5): `8 / -3` -> `-2`"""
        e = Expression("/", [Primary(target=8), Primary(target=-3)])

        self.assertEqual(str(e), "8 / -3")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Primary(target=-2))
        self.assertEqual(str(result), "-2")
        verify_structure_integrity(self, result)

    def test_e9(self):
        """E9(R6, R7): `x * x / -y` -> `-(x * x / y)`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("/", [
            Expression("*", [Primary(target=VariableRef(x))] * 2),
            Primary(sign="-", target=VariableRef(y))
        ])

        self.assertEqual(str(e), "x * x / -y")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Primary(sign="-", target=Expression("/", [
            Expression("*", [Primary(target=VariableRef(x))] * 2),
            Primary(target=VariableRef(y))
        ])))
        self.assertEqual(str(result), "-(x * x / y)")
        verify_structure_integrity(self, result)

    def test_e10(self):
        """E10(R6, R7): `-(x * x) / -y` -> `(x * x) / y`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("/", [
            Primary(sign="-", target=Expression("*", [Primary(target=VariableRef(x))] * 2)),
            Primary(sign="-", target=VariableRef(y))
        ])

        self.assertEqual(str(e), "-(x * x) / -y")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Expression("/", [
            Expression("*", [Primary(target=VariableRef(x))] * 2),
            Primary(target=VariableRef(y))
        ]))
        self.assertEqual(str(result), "(x * x) / y")
        verify_structure_integrity(self, result)

    def test_e11(self):
        """E11: `x * x * x / y` -> `x ** 3 / y`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("/", [
            Expression("*", [Primary(target=VariableRef(x))] * 3),
            Primary(target=VariableRef(y))
        ])

        self.assertEqual(str(e), "x * x * x / y")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Expression("/", [
            Expression("*", [Primary(target=VariableRef(x))] * 3),
            Primary(target=VariableRef(y))
        ]))
        self.assertEqual(str(result), "x * x * x / y")
        verify_structure_integrity(self, result)

    @skip("The associated behavior has not yet been implemented.")
    def test_e12(self):
        """E12(R9): `x * x * y / y * x` -> `x`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("/", [
            Expression("*", [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(y))
            ]),
            Expression("*", [
                Primary(target=VariableRef(y)),
                Primary(target=VariableRef(x))
            ]),
        ])

        self.assertEqual(str(e), "x * x * y / y * x")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Primary(target=VariableRef(x)))
        self.assertEqual(str(result), "x")
        verify_structure_integrity(self, result)

    @skip("The associated behavior has not yet been implemented.")
    def test_e13(self):
        """E13(R9): `x ** 2 * y * z / x * y` -> `x * z`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        z = Variable("z", _type=Type("Integer", 0))
        e = Expression("/", [
            Expression("*", [
                Expression("**", [Primary(target=VariableRef(x)), Primary(target=2)]),
                Primary(target=VariableRef(y)),
                Primary(target=VariableRef(z))
            ]),
            Expression("*", [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(y))
            ]),
        ])

        self.assertEqual(str(e), "x ** 2 * y * z / x * y")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Expression("*", [
            Primary(target=VariableRef(x)),
            Primary(target=VariableRef(y))
        ]))
        self.assertEqual(str(result), "x * y")
        verify_structure_integrity(self, result)

    @skip("The associated behavior has not yet been implemented.")
    def test_e14(self):
        """E14(R6, R7, R9): `-(8 * x * (y + 1) * z) / x * z` -> `-(8 * (y + 1))`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        z = Variable("z", _type=Type("Integer", 0))
        e = Expression("/", [
            Primary(sign="-", target=Expression("*", [
                Primary(target=8),
                Primary(target=VariableRef(x)),
                Primary(target=Expression("+", [Primary(target=VariableRef(y)), Primary(target=1)])),
                Primary(target=VariableRef(z))
            ])),
            Expression("*", [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(z))
            ]),
        ])

        self.assertEqual(str(e), "-(8 * x * (y + 1) * z) / x * z")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Primary(sign="-", target=Expression("*", [
            Primary(target=8),
            Primary(target=Expression("+", [Primary(target=VariableRef(y)), Primary(target=1)])),
        ])))
        self.assertEqual(str(result), "-(8 * (y + 1))")
        verify_structure_integrity(self, result)

    @skip("The associated behavior has not yet been implemented.")
    def test_e15(self):
        """E15(R9): `x * y / (x * z)` -> `y / z`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        z = Variable("z", _type=Type("Integer", 0))
        e = Expression("/", [
            Expression("*", [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(y))
            ]),
            Primary(target=Expression("*", [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(z))
            ])),
        ])

        self.assertEqual(str(e), "x * y / (x * z)")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Expression("/", [
            Primary(target=VariableRef(y)),
            Primary(target=VariableRef(z))
        ]))
        self.assertEqual(str(result), "y / z")
        verify_structure_integrity(self, result)

    def test_e16(self):
        """E15(R8): `0 / x` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("/", [
            Primary(target=0),
            Primary(target=VariableRef(x))
        ])

        self.assertEqual(str(e), "0 / x")
        verify_simplification_assumption(self, e)
        result = simplify_division(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)


class TestSimplifyModulo(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_e1(self):
        """E1(R1): `x % 0` -> `ZeroDivisionError()`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("%", [Primary(target=VariableRef(x)), Primary(target=0)])

        self.assertEqual(str(e), "x % 0")
        self.assertRaises(ZeroDivisionError, simplify_modulo, e)

    def test_e2(self):
        """E2(A2): `x % x % x` -> `Exception()`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("%", [Primary(target=VariableRef(x))] * 3)

        self.assertEqual(str(e), "x % x % x")
        self.assertRaises(Exception, simplify_modulo, e)

    def test_e3(self):
        """E3(R2): `x ** y % 1` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("%", [Expression("**", [
            Primary(target=VariableRef(x)),
            Primary(target=VariableRef(y))
        ]), Primary(target=1)])

        self.assertEqual(str(e), "x ** y % 1")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e4(self):
        """E4(R3): `x * y % (y * x)` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("%", [
            Expression("*", [Primary(target=VariableRef(x)), Primary(target=VariableRef(y))]),
            Primary(target=Expression("*", [Primary(target=VariableRef(y)), Primary(target=VariableRef(x))]))
        ])

        self.assertEqual(str(e), "x * y % (y * x)")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e5(self):
        """E5(R4): `8 % 3` -> `2`"""
        e = Expression("%", [Primary(target=8), Primary(target=3)])

        self.assertEqual(str(e), "8 % 3")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=2))
        self.assertEqual(str(result), "2")
        verify_structure_integrity(self, result)

    def test_e6(self):
        """E6(R4): `8 % -3` -> `-1`"""
        e = Expression("%", [Primary(target=8), Primary(target=-3)])

        self.assertEqual(str(e), "8 % -3")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=-1))
        self.assertEqual(str(result), "-1")
        verify_structure_integrity(self, result)

    def test_e7_1(self):
        """E7: `2 * x % y` -> `2 * x % y`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("%", [
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]),
            Primary(target=VariableRef(y))
        ])

        self.assertEqual(str(e), "2 * x % y")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Expression("%", [
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))]),
            Primary(target=VariableRef(y))
        ]))
        self.assertEqual(str(result), "2 * x % y")
        verify_structure_integrity(self, result)

    def test_e7_2(self):
        """E7: `3 * x % 2 * x` -> `3 * x % 2 * x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("%", [
            Expression("*", [Primary(target=3), Primary(target=VariableRef(x))]),
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))])
        ])

        self.assertEqual(str(e), "3 * x % 2 * x")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Expression("%", [
            Expression("*", [Primary(target=3), Primary(target=VariableRef(x))]),
            Expression("*", [Primary(target=2), Primary(target=VariableRef(x))])
        ]))
        self.assertEqual(str(result), "3 * x % 2 * x")
        verify_structure_integrity(self, result)

    def test_e8(self):
        """E8(R5): `x * y % x` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("%", [
            Expression("*", [Primary(target=VariableRef(x)), Primary(target=VariableRef(y))]),
            Primary(target=VariableRef(x))
        ])

        self.assertEqual(str(e), "x * y % x")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e9_1(self):
        """E9(R5): `(x + 1) * y * z % -(x + 1) * z` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        z = Variable("z", _type=Type("Integer", 0))
        e = Expression("%", [
            Expression("*", [
                Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
                Primary(target=VariableRef(y)),
                Primary(target=VariableRef(z))
            ]),
            Primary(sign="-", target=Expression("*", [
                Primary(target=Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)])),
                Primary(target=VariableRef(z))
            ])),
        ])

        self.assertEqual(str(e), "(x + 1) * y * z % -((x + 1) * z)")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e9_2(self):
        """E9(R5): `x * x * y * z % -(x * x * z)` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        z = Variable("z", _type=Type("Integer", 0))
        e = Expression("%", [
            Expression("*", [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(y)),
                Primary(target=VariableRef(z))
            ]),
            Primary(sign="-", target=Expression("*", [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(z))
            ])),
        ])

        self.assertEqual(str(e), "x * x * y * z % -(x * x * z)")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e10_1(self):
        """E10(R5): `-(x * x) % -x` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("%", [
            Primary(sign="-", target=Expression("*", [Primary(target=VariableRef(x))] * 2)),
            Primary(sign="-", target=VariableRef(x))
        ])

        self.assertEqual(str(e), "-(x * x) % -x")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e10_2(self):
        """E10(R5): `x ** y * x ** y % -(x ** y)` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("%", [
            Expression("*", [Expression("**", [
                Primary(target=VariableRef(x)), Primary(target=VariableRef(y))
            ])] * 2),
            Primary(sign="-", target=Expression("**", [
                Primary(target=VariableRef(x)), Primary(target=VariableRef(y))
            ]))
        ])

        self.assertEqual(str(e), "x ** y * x ** y % -(x ** y)")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e11_1(self):
        """E11(R5): `4 * x * x % -(2 * x)` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("%", [
            Expression("*", [
                Primary(target=4),
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(x))
            ]),
            Primary(sign="-", target=Primary(target=Expression("*", [
                Primary(target=2),
                Primary(target=VariableRef(x))
            ])))
        ])

        self.assertEqual(str(e), "4 * x * x % -(2 * x)")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e11_2(self):
        """E11(R5): `-(4 * x * x) % 2` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("%", [
            Primary(sign="-", target=Expression("*", [
                Primary(target=4),
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(x))
            ])),
            Primary(target=2)
        ])

        self.assertEqual(str(e), "-(4 * x * x) % 2")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)

    def test_e12(self):
        """E12(R6): `0 % x` -> `0`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("%", [Primary(target=0), Primary(target=VariableRef(x))])

        self.assertEqual(str(e), "0 % x")
        verify_simplification_assumption(self, e)
        result = simplify_modulo(e)

        self.assertEqual(result, Primary(target=0))
        self.assertEqual(str(result), "0")
        verify_structure_integrity(self, result)


class TestSimplifyPower(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_e1(self):
        """E1(A2): `x ** x ** x` -> `Exception()`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("**", [Primary(target=VariableRef(x))] * 3)

        self.assertEqual(str(e), "x ** x ** x")
        self.assertRaises(Exception, simplify_power, e)

    def test_e2(self):
        """E2(R3): `x ** -1` -> `Exception()`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("**", [Primary(target=VariableRef(x)), Primary(target=-1)])

        self.assertEqual(str(e), "x ** -1")
        self.assertRaises(Exception, simplify_power, e)

    def test_e3(self):
        """E3(R1): `-x ** 0` -> `1`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("**", [
            Primary(sign="-", target=Primary(target=VariableRef(x))),
            Primary(target=0)
        ])

        self.assertEqual(str(e), "-x ** 0")
        verify_simplification_assumption(self, e)
        result = simplify_power(e)

        self.assertEqual(result, Primary(target=1))
        self.assertEqual(str(result), "1")
        verify_structure_integrity(self, result)

    def test_e4(self):
        """E4(R2): `x ** 1` -> `x`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("**", [Primary(target=VariableRef(x)), Primary(target=1)])

        self.assertEqual(str(e), "x ** 1")
        verify_simplification_assumption(self, e)
        result = simplify_power(e)

        self.assertEqual(result, Primary(target=VariableRef(x)))
        self.assertEqual(str(result), "x")
        verify_structure_integrity(self, result)

    def test_e5(self):
        """E6(R4): `-2 ** 4` -> `16`"""
        e = Expression("**", [Primary(target=-2), Primary(target=4)])

        self.assertEqual(str(e), "-2 ** 4")
        verify_simplification_assumption(self, e)
        result = simplify_power(e)

        self.assertEqual(result, Primary(target=16))
        self.assertEqual(str(result), "16")
        verify_structure_integrity(self, result)

    def test_e6(self):
        """E6(R4): `-2 ** 3` -> `-8`"""
        e = Expression("**", [Primary(target=-2), Primary(target=3)])

        self.assertEqual(str(e), "-2 ** 3")
        verify_simplification_assumption(self, e)
        result = simplify_power(e)

        self.assertEqual(result, Primary(target=-8))
        self.assertEqual(str(result), "-8")
        verify_structure_integrity(self, result)

    def test_e7(self):
        """E7(R5): `-(x % 2) ** 3` -> `-((x % 2) * (x % 2) * (x % 2))`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression("**", [
            Primary(sign="-", target=Expression("%", [
                Primary(target=VariableRef(x)),
                Primary(target=2)
            ])),
            Primary(target=3)
        ])

        self.assertEqual(str(e), "-(x % 2) ** 3")
        verify_simplification_assumption(self, e)
        result = simplify_power(e)

        self.assertEqual(result, Primary(sign="-", target=Expression("*", [
            Primary(target=Expression("%", [Primary(target=VariableRef(x)), Primary(target=2)])),
        ] * 3)))
        self.assertEqual(str(result), "-((x % 2) * (x % 2) * (x % 2))")
        verify_structure_integrity(self, result)

    def test_e8(self):
        """E8: `x ** y` -> `x ** y`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        e = Expression("**", [Primary(target=VariableRef(x)), Primary(target=VariableRef(y))])

        self.assertEqual(str(e), "x ** y")
        verify_simplification_assumption(self, e)
        result = simplify_power(e)

        self.assertEqual(result, Expression("**", [Primary(target=VariableRef(x)), Primary(target=VariableRef(y))]))
        self.assertEqual(str(result), "x ** y")
        verify_structure_integrity(self, result)


class TestSimplifyComparison(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_e1(self):
        """E1(A2): `x` -> `Exception()`"""
        x = Variable("x", _type=Type("Integer", 0))
        e = Expression(">", [Primary(target=VariableRef(x))])

        self.assertEqual(str(e), "x")
        self.assertRaises(Exception, simplify_comparison, e)

    def test_e2(self):
        """E2(R1): `x {op} y {op} z` -> `x {op} y and y {op} z`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))
        z = Variable("z", _type=Type("Integer", 0))

        operators = [">", "<", ">=", "<=", "=", "!="]
        for op in operators:
            e = Expression(op, [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(y)),
                Primary(target=VariableRef(z))
            ])

            self.assertEqual(str(e), f"x {op} y {op} z")
            verify_simplification_assumption(self, e)
            result = simplify_comparison(e)
    
            self.assertEqual(result, Expression("and", [
                Expression(op, [
                    Primary(target=VariableRef(x)),
                    Primary(target=VariableRef(y))
                ]),
                Expression(op, [
                    Primary(target=VariableRef(y)),
                    Primary(target=VariableRef(z))
                ])
            ]))
            self.assertEqual(str(result), f"x {op} y and y {op} z")
            verify_structure_integrity(self, result)

    def test_e3(self):
        """E3(R1): `x {op} x {op} x + 1` -> `false`"""
        x = Variable("x", _type=Type("Integer", 0))

        operators = [">", "<", ">=", "=", "!="]
        for op in operators:
            e = Expression(op, [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(x)),
                Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)]),
            ])

            self.assertEqual(str(e), f"x {op} x {op} x + 1")
            verify_simplification_assumption(self, e)
            result = simplify_comparison(e)
    
            self.assertEqual(result, Primary(target=False))
            self.assertEqual(str(result), "false")
            verify_structure_integrity(self, result)

    def test_e4(self):
        """E4(R2): `x < x + 1` -> `true`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("<", [
            Primary(target=VariableRef(x)),
            Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)]),
        ])

        self.assertEqual(str(e), "x < x + 1")
        verify_simplification_assumption(self, e)
        result = simplify_comparison(e)

        self.assertEqual(result, Primary(target=True))
        self.assertEqual(str(result), "true")
        verify_structure_integrity(self, result)

    def test_e5(self):
        """E5(R3): `x = x + 1` -> `false`"""
        x = Variable("x", _type=Type("Integer", 0))

        e = Expression("=", [
            Primary(target=VariableRef(x)),
            Expression("+", [Primary(target=VariableRef(x)), Primary(target=1)]),
        ])

        self.assertEqual(str(e), "x = x + 1")
        verify_simplification_assumption(self, e)
        result = simplify_comparison(e)

        self.assertEqual(result, Primary(target=False))
        self.assertEqual(str(result), "false")
        verify_structure_integrity(self, result)

    def test_e6(self):
        """E6(R5): `-8 < 3` -> `true`"""
        e = Expression("<", [
            Primary(target=-8),
            Primary(target=3)
        ])

        self.assertEqual(str(e), "-8 < 3")
        verify_simplification_assumption(self, e)
        result = simplify_comparison(e)

        self.assertEqual(result, Primary(target=True))
        self.assertEqual(str(result), "true")
        verify_structure_integrity(self, result)

    def test_e7(self):
        """E7: `x {op} y` -> `x {op} y`"""
        x = Variable("x", _type=Type("Integer", 0))
        y = Variable("y", _type=Type("Integer", 0))

        operators = [">", "<", ">=", "<=", "=", "!="]
        for op in operators:
            e = Expression(op, [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(y))
            ])

            self.assertEqual(str(e), f"x {op} y")
            verify_simplification_assumption(self, e)
            result = simplify_comparison(e)
    
            self.assertEqual(result, Expression(op, [
                Primary(target=VariableRef(x)),
                Primary(target=VariableRef(y))
            ]))
            self.assertEqual(str(result), f"x {op} y")
            verify_structure_integrity(self, result)







@skip("")
class TestSimplifyExpression(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_simplify_expression_exception(self):
        for op in ["**", "%", "/"]:
            e = Expression(op, [Primary(target=5)]*3)
            self.assertRaises(Exception, simplify, e)

    def test_simplify_expression_1(self):
        v = Variable("x", _type=Type("Integer", 0))
        e = Expression(
            "+", [Expression("-", [Primary(target=5), Primary(target=2)]), Primary(target=VariableRef(var=v))]
        )
        result = simplify(e)

        self.assertEqual(result, Expression("+", [Primary(target=VariableRef(var=v)), Primary(target=3)]))

    def test_simplify_expression_2(self):
        e = Expression("+", [
            Expression("-", [Primary(target=5), Primary(target=2)]),
            Expression("*", [Primary(target=3), Primary(target=4)])
        ])
        result = simplify(e)

        self.assertEqual(result, Primary(target=15))

    def test_simplify_expression_empty_1(self):
        e = Expression("", [Primary(target=4)])
        result = simplify(e)

        self.assertEqual(result, Primary(target=4))

    def test_simplify_expression_merge_1(self):
        e = Expression("+", [math_expr_dummy(), Expression("+", [math_expr_dummy(), math_expr_dummy()])])
        result = simplify(e)

        self.assertEqual(result, Expression("+", [math_expr_dummy()] * 3))

    def test_simplify_expression_merge_2(self):
        e = Expression("+", [math_expr_dummy(), Primary(target=Expression("+", [math_expr_dummy(), math_expr_dummy()]))])
        result = simplify(e)

        self.assertEqual(result, Expression("+", [math_expr_dummy()] * 3))

    def test_simplify_expression_merge_3(self):
        e = Expression("+", [
            math_expr_dummy(),
            Primary(sign="-", target=Expression("+", [math_expr_dummy(), math_expr_dummy()]))
        ])
        result = simplify(e)

        self.assertEqual(result, Expression(
            "+", [
                math_expr_dummy(),
                Primary(sign="-", target=math_expr_dummy()),
                Primary(sign="-", target=math_expr_dummy())
            ]
        ))

    def test_simplify_expression_merge_4(self):
        e = Expression("**", [
            math_expr_dummy(),
            Expression("**", [
                math_expr_dummy(),
                Expression("**", [math_expr_dummy(), math_expr_dummy()])
            ])
        ])
        result = simplify(e)

        self.assertEqual(result, Expression("**", [
            math_expr_dummy(),
            Expression("**", [
                math_expr_dummy(),
                Expression("**", [math_expr_dummy(), math_expr_dummy()])
            ])
        ]))

    def test_simplify_expression_practical_error_1(self):
        # The expression i - 2 - 2 was resolved erroneously to i due to premature simplification.
        v = Variable("i", _type=Type("Integer", 0))
        e = Expression("-", [Primary(target=VariableRef(var=v)), Primary(target=2), Primary(target=2)])
        result = simplify(e)

        self.assertEqual(result, Expression("+", [Primary(target=VariableRef(var=v)), Primary(target=-4)]))

    def test_simplify_expression_practical_error_2(self):
        # The expression i - (3 - 3) was resolved erroneously due to an implementation error.
        v = Variable("i", _type=Type("Integer", 0))
        e = Expression("-", [
            Primary(target=VariableRef(var=v)),
            Primary(target=Expression("-", [Primary(target=3), Primary(target=3)]))
        ])
        result = simplify(e)

        self.assertEqual(result, Primary(target=VariableRef(var=v)))


@skip("")
class TestSimplifyPrimary(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_simplify_primary_basic_values_1(self):
        p = Primary(target=Expression("*", [Primary(target=3), Primary(target=4)]))
        result = simplify(p)

        self.assertEqual(result, Primary(target=12))
        self.assertEqual(str(result), "12")

    def test_simplify_primary_basic_values_2(self):
        p = Primary(sign="-", target=Expression("*", [Primary(target=3), Primary(target=4)]))
        result = simplify(p)

        self.assertEqual(result, Primary(target=-12))
        self.assertEqual(str(result), "-12")

    def test_simplify_primary_basic_values_3(self):
        p = Primary(sign="-", target=Expression("*", [Primary(target=-3), Primary(target=4)]))
        result = simplify(p)

        self.assertEqual(result, Primary(target=12))
        self.assertEqual(str(result), "12")

    def test_simplify_primary_basic_values_4(self):
        p = Primary(sign="not", target=Expression("xor", [Primary(target=True), Primary(target=False)]))
        result = simplify(p)

        self.assertEqual(result, Primary(target=False))
        self.assertEqual(str(result), "false")

    def test_simplify_primary_ref_1(self):
        v = Variable("x", _type=Type("Integer", 0))
        p = Primary(sign="-", target=Expression("*", [Primary(target=3), Primary(target=VariableRef(var=v))]))
        result = simplify(p)

        self.assertEqual(
            result, Primary(sign="-", target=Expression("*", [Primary(target=3), Primary(target=VariableRef(var=v))]))
        )
        self.assertEqual(str(result), "-(3 * x)")

    def test_simplify_primary_value_double_negation_1(self):
        p2 = Primary(sign="-", target=Primary(sign="-", target=5))
        result = simplify(p2)

        self.assertEqual(result, Primary(target=5))
        self.assertEqual(str(result), "5")

    def test_simplify_primary_ref_double_negation_1(self):
        v = Variable("x", _type=Type("Boolean", 0))
        p = Primary(sign="not", target=Primary(sign="not", target=VariableRef(var=v)))
        result = simplify(p)

        self.assertEqual(result, Primary(target=VariableRef(var=v)))
        self.assertEqual(str(result), "x")

    def test_simplify_primary_body_double_negation_1(self):
        v = Variable("x", _type=Type("Integer", 0))
        p = Primary(sign="-", target=Primary(
            sign="-", target=Expression("*", [Primary(target=VariableRef(var=v)), Primary(target=VariableRef(var=v))])
        ))
        result = simplify(p)

        self.assertEqual(
            result, Expression("*", [Primary(target=VariableRef(var=v)), Primary(target=VariableRef(var=v))])
        )
        self.assertEqual(str(result), "(x * x)")

    def test_simplify_primary_superfluous_nested_1(self):
        v = Variable("x", _type=Type("Integer", 0))
        p = Primary(sign="", target=Primary(
            sign="-", target=Expression("*", [Primary(target=VariableRef(var=v)), Primary(target=VariableRef(var=v))])
        ))
        result = simplify(p)

        self.assertEqual(result, Primary(
            sign="-", target=Expression("*", [Primary(target=VariableRef(var=v)), Primary(target=VariableRef(var=v))])
        ))
        self.assertEqual(str(result), "-(x * x)")

    def test_simplify_primary_superfluous_nested_2(self):
        v = Variable("x", _type=Type("Integer", 0))
        p = Primary(sign="-", target=Primary(
            sign="", target=Expression("*", [Primary(target=VariableRef(var=v)), Primary(target=VariableRef(var=v))])
        ))
        result = simplify(p)

        self.assertEqual(result, Primary(
            sign="-", target=Expression("*", [Primary(target=VariableRef(var=v)), Primary(target=VariableRef(var=v))])
        ))
        self.assertEqual(str(result), "-(x * x)")

    def test_simplify_primary_superfluous_nested_3(self):
        v = Variable("x", _type=Type("Integer", 0))
        p = Primary(sign="-", target=Primary(sign="", target=VariableRef(var=v)))
        result = simplify(p)

        self.assertEqual(result, Primary(sign="-", target=VariableRef(var=v)))
        self.assertEqual(str(result), "-x")

    def test_simplify_primary_superfluous_nested_4(self):
        v = Variable("x", _type=Type("Integer", 0))
        p = Primary(sign="", target=Primary(sign="-", target=VariableRef(var=v)))
        result = simplify(p)

        self.assertEqual(result, Primary(sign="-", target=VariableRef(var=v)))
        self.assertEqual(str(result), "-x")

    def test_simplify_primary_addition_negation_2(self):
        v = Variable("x", _type=Type("Integer", 0))
        for cls in [Assignment, Composite, Transition, VariableRef]:
            p = Primary(sign="-", target=Expression("+", [Primary(target=VariableRef(var=v)), Primary(target=5)]))
            p.parent = cls()
            result = simplify(p)

            self.assertEqual(result, Expression(
                "+", [Primary(sign="-", target=VariableRef(var=v)), Primary(sign="-", target=5)]
            ))
            self.assertEqual(str(result), "(-x + -5)")

    def test_simplify_primary_addition_negation_3(self):
        v = Variable("x", _type=Type("Integer", 0))
        for cls in [Assignment, Composite, Transition, VariableRef]:
            p = Primary(sign="-", target=Expression("+", [Primary(target=VariableRef(var=v)), math_expr_dummy()]))
            p.parent = cls()
            result = simplify(p)

            self.assertEqual(result, Expression(
                "+", [Primary(sign="-", target=VariableRef(var=v)), Primary(sign="-", target=math_expr_dummy())]
            ))
            self.assertEqual(str(result), "(-x + -(w * v))")

    def test_simplify_primary_mandatory_brackets_1(self):
        v = Variable("x", _type=Type("Integer", 0))
        p = Primary(sign="-", target=Primary(sign="-", target=Expression("+", [Primary(target=VariableRef(v))]*2)))
        e = Expression("*", [Primary(target=10), p])
        result = simplify(e)

        self.assertEqual(result, Expression("*", [
            Primary(target=10),
            Primary(target=Expression("+", [Primary(target=VariableRef(v))]*2))
        ]))
        self.assertEqual(str(result), "10 * (x + x)")


@skip("")
class TestSimplifyVariableRef(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_simplify_variable_ref_1(self):
        v = Variable("x", _type=Type("Integer", 10))
        vr = VariableRef(var=v, index=Expression("+", [Primary(target=5), Primary(target=2)]))
        result = simplify(vr)

        self.assertEqual(result, VariableRef(var=v, index=Primary(target=7)))

    def test_simplify_variable_ref_2(self):
        v = Variable("x", _type=Type("Integer", 0))
        vr = VariableRef(var=v)
        result = simplify(vr)

        self.assertEqual(result, VariableRef(var=v))


@skip("")
class TestSimplifyAssignment(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_simplify_assignment_1(self):
        v = Variable("x", _type=Type("Integer", 3))
        a = Assignment(
            left=VariableRef(
                var=v,
                index=Expression("+", [Primary(target=1), Primary(target=1)])
            ),
            right=Primary(sign="-", target=Primary(sign="-", target=VariableRef(var=v, index=Primary(target=1))))
        )
        result = simplify(a)

        self.assertEqual(result, Assignment(
            left=VariableRef(var=v, index=Primary(target=2)),
            right=Primary(target=VariableRef(var=v, index=Primary(target=1)))
        ))


@skip("")
class TestSimplifyComposite(TestCase):
    def setUp(self):
        smt.clear_smt_cache()

    def test_simplify_composite_1(self):
        v = Variable("b", _type=Type("Boolean", 2))
        v2 = Variable("x", _type=Type("Integer", 2))
        c = Composite(
            guard=VariableRef(
                var=v,
                index=Expression("+", [Primary(target=1), Primary(target=1)])
            ),
            assignments=[
                Assignment(
                    left=VariableRef(var=v2, index=Primary(target=0)),
                    right=Primary(sign="-", target=Primary(sign="-", target=VariableRef(var=v2)))
                )
            ]
        )
        result = simplify(c)

        self.assertEqual(result, Composite(
            guard=VariableRef(
                var=v,
                index=Primary(target=2)
            ),
            assignments=[
                Assignment(
                    left=VariableRef(var=v2, index=Primary(target=0)),
                    right=Primary(target=VariableRef(var=v2))
                )
            ]
        ))

    def test_simplify_composite_2(self):
        v = Variable("x", _type=Type("Integer", 2))
        c = Composite(
            assignments=[
                Assignment(
                    left=VariableRef(var=v, index=Primary(target=0)),
                    right=Primary(
                        sign="-",
                        target=Primary(sign="-", target=VariableRef(var=v, index=Primary(target=1)))
                    )
                )
            ]
        )
        result = simplify(c)

        self.assertEqual(result, Composite(
            guard=Primary(target=True),
            assignments=[
                Assignment(
                    left=VariableRef(var=v, index=Primary(target=0)),
                    right=Primary(target=VariableRef(var=v, index=Primary(target=1)))
                )
            ]
        ))
