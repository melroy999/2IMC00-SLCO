from __future__ import annotations
from unittest import TestCase

from z3 import z3

import objects.ast.models as models

from rendering.util.to_smt import n_ary_to_binary_operations, variable_ref_to_smt, primary_to_smt, variable_to_smt, \
    composite_to_smt


class TestIsTrue(TestCase):
    def test_integer_division_1(self):
        self.assertTrue(models.Expression("=", [
            models.Expression("/", [models.Primary(target=1), models.Primary(target=3)]),
            models.Primary(target=0)
        ]).is_true())

    def test_integer_division_2(self):
        self.assertTrue(models.Expression("=", [
            models.Expression("/", [models.Primary(target=4), models.Primary(target=3)]),
            models.Primary(target=1)
        ]).is_true())


class TestToBinary(TestCase):
    def test_n_ary_to_binary_operations_basic_1(self):
        e = models.Expression("+")
        result = n_ary_to_binary_operations(e, [5, 1, 3, 9])

        self.assertEqual(result, 5 + 1 + 3 + 9)

    def test_n_ary_to_binary_operations_basic_2(self):
        e = models.Expression("-")
        result = n_ary_to_binary_operations(e, [5, 1, 3, 9])

        self.assertEqual(result, 5 - 1 - 3 - 9)

    def test_n_ary_to_binary_operations_complex_1(self):
        x = z3.Int("x")
        y = z3.Int("y")
        z = z3.Int("z")
        e = models.Expression("+")
        result = n_ary_to_binary_operations(e, [x, y, z])

        self.assertEqual(result, x + y + z)

    def test_n_ary_to_binary_operations_complex_2(self):
        x = z3.Int("x")
        y = z3.Int("y")
        z = z3.Int("z")
        e = models.Expression("-")
        result = n_ary_to_binary_operations(e, [x, y, z])

        self.assertEqual(result, x - y - z)

    def test_n_ary_to_binary_associativity_rules_math(self):
        x = 2
        y = 3
        z = 4
        i = 5

        e = models.Expression("**")
        result = n_ary_to_binary_operations(e, [i, z, y, x])
        self.assertEqual(result, i ** z ** y ** x)

        e = models.Expression("+")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x + y + z + i)

        e = models.Expression("-")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x - y - z - i)

        e = models.Expression("*")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x * y * z * i)

        x = 200
        y = 10
        z = 5
        i = 2

        e = models.Expression("/")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x / y / z / i)

        x = 13
        y = 8
        z = 5
        i = 3

        e = models.Expression("%")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x % y % z % i)

    def test_n_ary_to_binary_associativity_rules_comparative_logic(self):
        x = 2
        y = 3
        z = 4
        i = 5

        e = models.Expression("<")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x < y < z < i)

        e = models.Expression("!=")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x != y != z != i)

        x = 2
        y = 2
        z = 2
        i = 2

        e = models.Expression("<=")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x <= y <= z <= i)

        e = models.Expression(">=")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x >= y >= z >= i)

        e = models.Expression("=")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x == y == z == i)

        x = 5
        y = 4
        z = 3
        i = 2

        e = models.Expression(">")
        result = n_ary_to_binary_operations(e, [x, y, z, i])
        self.assertEqual(result, x > y > z > i)

    def test_n_ary_to_binary_exception(self):
        e = models.Expression(">")
        self.assertRaises(Exception, n_ary_to_binary_operations, e, [])


class TestVariableRef(TestCase):
    def test_variable_ref_to_smt_1(self):
        v = models.Variable("i", _type=models.Type("Integer", 0))
        vr = models.VariableRef(var=v)
        result = variable_ref_to_smt(vr)

        self.assertEqual(str(result), "i")

    def test_variable_ref_to_smt_2(self):
        v = models.Variable("i", _type=models.Type("Integer", 2))
        vr = models.VariableRef(var=v, index=models.Primary(value=1))
        result = variable_ref_to_smt(vr)

        self.assertEqual(str(result), "i[1]")


class TestVariable(TestCase):
    def test_variable_to_smt_1(self):
        v = models.Variable("i", _type=models.Type("Integer", 0))
        result = variable_to_smt(v)

        self.assertIsInstance(result, z3.ArithRef)

    def test_variable_to_smt_2(self):
        v = models.Variable("i", _type=models.Type("Integer", 2))
        result = variable_to_smt(v)

        self.assertIsInstance(result, z3.ArrayRef)
        self.assertEqual(result, z3.Array("i", z3.IntSort(), z3.IntSort()))

    def test_variable_to_smt_3(self):
        v = models.Variable("i", _type=models.Type("Boolean", 0))
        result = variable_to_smt(v)

        self.assertIsInstance(result, z3.BoolRef)

    def test_variable_to_smt_4(self):
        v = models.Variable("i", _type=models.Type("Boolean", 2))
        result = variable_to_smt(v)

        self.assertIsInstance(result, z3.ArrayRef)
        self.assertEqual(result, z3.Array("i", z3.IntSort(), z3.BoolSort()))


class TestComposite(TestCase):
    def test_composite_to_smt_value_1(self):
        v = models.Variable("i", _type=models.Type("Integer", 0))
        vr = models.VariableRef(var=v)
        e = models.Expression("+")
        e.values = [vr, vr]
        c = models.Composite()
        c.guard = e
        result = composite_to_smt(c)

        self.assertEqual(str(result), "i + i")


class TestPrimary(TestCase):
    def test_primary_to_smt_value_1(self):
        p = models.Primary(value=True)
        result = primary_to_smt(p)

        self.assertEqual(str(result), "True")

    def test_primary_to_smt_value_2(self):
        p = models.Primary(value=5)
        result = primary_to_smt(p)

        self.assertEqual(str(result), "5")

    def test_primary_to_smt_value_3(self):
        p = models.Primary(sign="not", value=True)
        result = primary_to_smt(p)

        self.assertEqual(str(result), "Not(True)")

    def test_primary_to_smt_value_4(self):
        p = models.Primary(sign="-", value=5)
        result = primary_to_smt(p)

        self.assertEqual(str(result), "-5")

    def test_primary_to_smt_1_exception(self):
        # noinspection Pymodels.TypeChecker
        p = models.Primary(value="illegal")
        self.assertRaises(Exception, primary_to_smt, p)

    def test_primary_to_smt_ref_1(self):
        v = models.Variable("i", _type=models.Type("Integer", 0))
        vr = models.VariableRef(var=v)
        p = models.Primary(sign="-", target=vr)
        result = primary_to_smt(p)

        self.assertEqual(str(result), "-i")

    def test_primary_to_smt_body_1(self):
        v = models.Variable("i", _type=models.Type("Integer", 0))
        vr = models.VariableRef(var=v)
        e = models.Expression("**")
        e.values = [models.Primary(value=2), vr]
        p = models.Primary(target=e)
        result = primary_to_smt(p)

        self.assertEqual(str(result), "2**i")
