from unittest import TestCase

from objects.ast.models import Expression, Transition, Composite, Assignment, Primary, VariableRef, Variable, Type


def validate_parent_child_relationship(case: TestCase, target):
    for o in target:
        if o is not None:
            case.assertIs(o.parent, target)
            validate_parent_child_relationship(case, o)


# TODO: test for the revised skip_superfluous_expression method


class SimplifiableSkipEmptyExpressionTest(TestCase):
    def test_skip_superfluous_expression_skipable_expression(self):
        """
        Test 'skip_superfluous_expression' with an empty expression present.

        1:Expression(op=Any)                      1:Expression(op=Any)
            2:Expression(op=None)         =>          3:Expression(op=Any)
                3:Expression(op=Any)
        """
        e1: Expression = Expression("+")

        e2: Expression = Expression("")
        e3: Expression = Expression("-")
        e2.values = [e3]
        e1.values = [e2]

        result = e1._skip_superfluous_expression(e2)
        e1.values = [result]

        self.assertIs(result, e3)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_expression_not_skipable_expression(self):
        """
        Test 'skip_superfluous_expression' without an empty expression present.

        1:Expression(op=Any)                      1:Expression(op=Any)
            2:Expression(op!=None)        =>          2:Expression(op!=None)
                3:Any(op=Any)                             3:Any(op=Any)
                4:Any(op=Any)                             4:Any(op=Any)
        """
        e1: Expression = Expression("+")

        e2: Expression = Expression("-")
        e3: Expression = Expression("+")
        e4: Expression = Expression("+")
        e2.values = [e3, e4]
        e1.values = [e2]

        result = e1._skip_superfluous_expression(e2)
        e1.values = [result]

        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_expression_not_skipable_primary(self):
        """
        Test 'skip_superfluous_expression' with a primary node as child value.

        1:Expression(op=Any)                      1:Expression(op=Any)
            2:Primary(Any)                =>          2:Primary(Any)
        """
        e1: Expression = Expression("+")

        e2: Primary = Primary(value=0)
        e1.values = [e2]

        result = e1._skip_superfluous_expression(e2)
        e1.values = [result]

        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_expression_not_skipable_composite(self):
        """
        Test 'skip_superfluous_expression' with a composite node as child value.

        1:Transition(Any)                       1:Transition(Any)
            2:Composite(Any)              =>        2:Composite(Any)
        """
        e1: Transition = Transition(0)

        e2: Composite = Composite()
        e1.statements = [e2]

        result = e1._skip_superfluous_expression(e2)
        e1.statements = [result]

        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_expression_not_skipable_assignment(self):
        """
        Test 'skip_superfluous_expression' with an assignment node as child value.

        1:Transition(Any)                      1:Transition(Any)
            2:Assignment(Any)             =>       2:Assignment(Any)
        """
        e1: Transition = Transition(0)

        e2: Assignment = Assignment()
        e1.statements = [e2]

        result = e1._skip_superfluous_expression(e2)
        e1.statements = [result]

        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)


# noinspection DuplicatedCode
class SimplifiableSkipSuperfluousPrimary(TestCase):
    def test_skip_superfluous_primary_skipable_nested_primary(self):
        """
        Test 'skip_superfluous_primary' with a superfluous primary present, with the body being another primary.

        1:Expression(op=Any)                      1:Expression(op=Any)
            2:Primary(sign="",body!=None) =>          3:Primary(op=Any)
                3:Primary(op=Any)
        """
        e1: Expression = Expression("+")

        e3: Primary = Primary(value=0)
        e2: Primary = Primary(body=e3)
        e1.values = [e2]

        result = e1._skip_superfluous_primary(e2)
        e1.values = [result]

        self.assertIs(result, e3)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_skipable_matching_nested_expression_n_ary(self):
        """
        Test 'skip_superfluous_primary' with a primary present, with matching expression operators.

        1:Expression(op=n-ary)                    1:Expression(op=n-ary)
            2:Primary(sign="",body!=None) =>          3:Expression(op=1.op)
                3:Expression(op=1.op)
        """
        e1: Expression = Expression("+")

        e3: Expression = Expression("+")
        e2: Primary = Primary(body=e3)
        e1.values = [e2]

        result = e1._skip_superfluous_primary(e2)
        e1.values = [result]

        self.assertIs(result, e3)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_not_skipable_matching_nested_expression_binary(self):
        """
        Test 'skip_superfluous_primary' with a primary present, with matching expression operators.

        1:Expression(op=binary)                   1:Expression(op=binary)
            2:Primary(sign="",body!=None) =>          2:Primary(sign="",body!=None)
                3:Expression(op=1.op)                     3:Expression(op=1.op)
        """
        e1: Expression = Expression("/")

        e3: Expression = Expression("/")
        e2: Primary = Primary(body=e3)
        e1.values = [e2]

        result = e1._skip_superfluous_primary(e2)
        e1.values = [result]

        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_not_skipable_distinct_nested_expression_n_ary(self):
        """
        Test 'skip_superfluous_primary' with a primary present, with distinct expression operators.

        1:Expression(op=n-ary)                    1:Expression(op=n-ary)
            2:Primary(sign="",body!=None) =>          2:Primary(sign="",body!=None)
                3:Expression(op!=1.op)                    3:Expression(op!=1.op)
        """
        e1: Expression = Expression("*")

        e3: Expression = Expression("+")
        e2: Primary = Primary(body=e3)
        e1.values = [e2]

        result = e1._skip_superfluous_primary(e2)
        e1.values = [result]

        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_not_skipable_distinct_nested_expression_binary(self):
        """
        Test 'skip_superfluous_primary' with a primary present, with matching expression operators.

        1:Expression(op=binary)                   1:Expression(op=binary)
            2:Primary(sign="",body!=None) =>          2:Primary(sign="",body!=None)
                3:Expression(op!=1.op)                    3:Expression(op!=1.op)
        """
        e1: Expression = Expression("/")

        e3: Expression = Expression("/")
        e2: Primary = Primary(body=e3)
        e1.values = [e2]

        result = e1._skip_superfluous_primary(e2)
        e1.values = [result]

        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_not_skipable_signed(self):
        """
        Test 'skip_superfluous_primary' with a signed primary present.

        1:Expression(op=n-ary)                    1:Expression(op=n-ary)
            2:Primary(sign="not",body!=None) =>       2:Primary(sign="not",body!=None)
                3:Expression(op=1.op)                     3:Expression(op=1.op)
        """
        e1: Expression = Expression("and")

        e3: Expression = Expression("and")
        e2: Primary = Primary(sign="not", body=e3)
        e1.values = [e2]

        result = e1._skip_superfluous_primary(e2)
        e1.values = [result]

        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_skipable_composite_parent(self):
        """
        Test 'skip_superfluous_primary' with a superfluous primary present in a composite statement.

        1:Composite(Any)                          1:Composite(Any)
            2:Primary(sign="",body!=None) =>          3:Expression(Any)
                3:Expression(Any)
        """
        e1: Composite = Composite()

        e3: Expression = Expression("+")
        e2: Primary = Primary(body=e3)
        e1.guard = e2

        result = e1.guard = e1._skip_superfluous_primary(e2)
        self.assertIs(result, e3)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_not_skipable_composite_parent_signed(self):
        """
        Test 'skip_superfluous_primary' with a signed primary present in a composite statement.

        1:Composite(Any)                          1:Composite(Any)
            2:Primary(sign="not",body!=None) =>       2:Primary(sign="not",body!=None)
                3:Expression(Any)                         3:Expression(Any)
        """
        e1: Composite = Composite()

        e3: Expression = Expression("and")
        e2: Primary = Primary(sign="not", body=e3)
        e1.guard = e2

        result = e1.guard = e1._skip_superfluous_primary(e2)
        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_skipable_assignment_parent(self):
        """
        Test 'skip_superfluous_primary' with a superfluous primary present in an assigment statement.

        1:Assignment(Any)                         1:Assignment(Any)
            2:Primary(sign="",body!=None) =>          3:Expression(Any)
                3:Expression(Any)
        """
        e1: Assignment = Assignment()

        e3: Expression = Expression("+")
        e2: Primary = Primary(body=e3)
        e1.right = e2

        result = e1.right = e1._skip_superfluous_primary(e2)
        self.assertIs(result, e3)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_not_skipable_assignment_parent_signed(self):
        """
        Test 'skip_superfluous_primary' with a signed primary present in a composite statement.

        1:Assignment(Any)                         1:Assignment(Any)
            2:Primary(sign="not",body!=None) =>       2:Primary(sign="not",body!=None)
                3:Expression(Any)                         3:Expression(Any)
        """
        e1: Assignment = Assignment()

        e3: Expression = Expression("and")
        e2: Primary = Primary(sign="not", body=e3)
        e1.right = e2

        result = e1.right = e1._skip_superfluous_primary(e2)
        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_skipable_variable_ref_parent(self):
        """
        Test 'skip_superfluous_primary' with a superfluous primary present in a variable reference statement.

        1:VariableRef(Any)                        1:VariableRef(Any)
            2:Primary(sign="",body!=None) =>          3:Expression(Any)
                3:Expression(Any)
        """
        e1: VariableRef = VariableRef()

        e3: Expression = Expression("+")
        e2: Primary = Primary(body=e3)
        e1.index = e2

        result = e1.index = e1._skip_superfluous_primary(e2)
        self.assertIs(result, e3)

        validate_parent_child_relationship(self, e1)

    def test_skip_superfluous_primary_skipable_variable_ref_parent_signed(self):
        """
        Test 'skip_superfluous_primary' with a signed primary present in a variable reference statement.

        1:VariableRef(Any)                        1:VariableRef(Any)
            2:Primary(sign="neg",body!=None) =>       2:Primary(sign="neg",body!=None)
                3:Expression(Any)                         3:Expression(Any)
        """
        e1: VariableRef = VariableRef()

        e3: Expression = Expression("-")
        e2: Primary = Primary(sign="neg", body=e3)
        e1.index = e2

        result = e1.index = e1._skip_superfluous_primary(e2)
        self.assertIs(result, e2)

        validate_parent_child_relationship(self, e1)


# noinspection DuplicatedCode
class SimplifiableSimplifyTransition(TestCase):
    def test_simplify_transition_empty(self):
        """
        Test the simplify method of the transition class, with the transition having no statements.
        1:Transition()                    =>      1:Transition()
                                                      2:Primary(value=True)
        """
        e1: Transition = Transition(0)

        e1.preprocess()

        self.assertEqual(len(e1.statements), 1)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertTrue(e1.statements[0].value)

        validate_parent_child_relationship(self, e1)

    def test_simplify_transition_single_false_expression(self):
        """
        Test the simplify method of the transition class, with the transition having a single false primary.
        1:Transition()                    =>      1:Transition()
            2:Expression(op=and)                      5:Primary(value=False)
                3:Primary(value=True)
                4:Primary(value=False)
        """
        e1: Transition = Transition(0)

        e2: Expression = Expression("and")
        e3: Primary = Primary(value=True)
        e4: Primary = Primary(value=False)
        e2.values = [e3, e4]
        e1.statements = [e2]

        e1.preprocess()

        self.assertEqual(len(e1.statements), 1)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertFalse(e1.statements[0].value)

        validate_parent_child_relationship(self, e1)

    def test_simplify_transition_single_false_primary(self):
        """
        Test the simplify method of the transition class, with the transition having a single false primary.
        1:Transition()                    =>      1:Transition()
            2:Primary(value=False)                    3:Primary(value=False)
        """
        e1: Transition = Transition(0)

        e2: Primary = Primary(value=False)
        e1.statements = [e2]

        e1.preprocess()

        self.assertEqual(len(e1.statements), 1)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertFalse(e1.statements[0].value)

        validate_parent_child_relationship(self, e1)

    def test_simplify_transition_single_true_primary(self):
        """
        Test the simplify method of the transition class, with the transition having a single true primary.
        1:Transition()                    =>      1:Transition()
            2:Primary(value=True)                     3:Primary(value=True)
        """
        e1: Transition = Transition(0)

        e2: Primary = Primary(value=True)
        e1.statements = [e2]

        e1.preprocess()

        self.assertEqual(len(e1.statements), 1)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertTrue(e1.statements[0].value)

        validate_parent_child_relationship(self, e1)

    def test_simplify_transition_single_false_composite(self):
        """
        Test the simplify method of the transition class, with the transition having one composite with a false guard.
        1:Transition()                    =>      1:Transition()
            2:Composite()                             4:Primary(value=False)
                3:Primary(value=False)
        """
        e1: Transition = Transition(0)

        e2: Composite = Composite()
        e3: Primary = Primary(value=False)
        e2.guard = e3
        e1.statements = [e2]

        e1.preprocess()

        self.assertEqual(len(e1.statements), 1)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertFalse(e1.statements[0].value)

        validate_parent_child_relationship(self, e1)

    def test_simplify_transition_true_single_true_composite_without_assignments(self):
        """
        Test the simplify method of the transition class, with the transition having one composite with a true guard.
        1:Transition()                    =>      1:Transition()
            2:Composite()                             6:Primary(value=True)
                3:Primary(value=True)
        """
        e1: Transition = Transition(0)

        e2: Composite = Composite()
        e3: Primary = Primary(value=True)
        e2.guard = e3
        e1.statements = [e2]

        e1.preprocess()

        self.assertEqual(len(e1.statements), 1)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertTrue(e1.statements[0].value)

        validate_parent_child_relationship(self, e1)

    def test_simplify_transition_true_single_true_composite_with_assignments(self):
        """
        Test the simplify method of the transition class, with the transition having one composite with a true guard.
        1:Transition()                    =>      1:Transition()
            2:Composite()                             6:Primary(value=True)
                3:Primary(value=True)                 2:Composite()
                4:Assignment()                            3:Primary(value=True)
                    v:VariableRef(name=x,type=Bool)       4:Assignment()
                    5:Primary(value=True)                     v:VariableRef(name=x,type=Bool)
                                                              5:Primary(value=True)
        """
        e1: Transition = Transition(0)

        v: Variable = Variable("x")
        v.type = Type("Boolean", 0)
        vr: VariableRef = VariableRef()
        vr.var = v

        e2: Composite = Composite()
        e3: Primary = Primary(value=True)
        e4: Assignment = Assignment()
        e5: Primary = Primary(value=True)
        e4.left = vr
        e4.right = e5
        e2.guard = e3
        e2.assignments = [e4]
        e1.statements = [e2]

        e1.preprocess()

        self.assertEqual(len(e1.statements), 2)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertTrue(e1.statements[0].value)

        self.assertTrue(isinstance(e1.statements[1], Composite))
        self.assertTrue(isinstance(e1.statements[1].guard, Primary))
        self.assertTrue(e1.statements[1].guard.value)
        self.assertIs(e1.statements[1], e2)

    def test_simplify_transition_true_primary_followed_by_false_primary(self):
        """
        Test the simplify method of the transition class, with the transition having a single true and false primary.
        1:Transition()                    =>      1:Transition()
            2:Primary(value=True)                     4:Primary(value=False)
            3:Primary(value=False)
        """
        e1: Transition = Transition(0)

        e2: Primary = Primary(value=True)
        e3: Primary = Primary(value=False)
        e1.statements = [e2, e3]

        e1.preprocess()

        self.assertEqual(len(e1.statements), 1)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertFalse(e1.statements[0].value)

        validate_parent_child_relationship(self, e1)

    def test_simplify_transition_true_expression_filtering(self):
        """
        Test the simplify method of the transition class, with several superfluous expressions evaluating to true.
        1:Transition()                    =>      1:Transition()
            2:Primary(value=True)                     9:Primary(value=True)
            3:Expression(op=or)                       6:Assignment()
                4:Primary(value=True)                     v:VariableRef(name=x,type=Bool)
                5:Primary(value=False)                    7:Primary(value=True)
            6:Assignment()                            8:Primary(sign=not,ref=x)
                v:VariableRef(name=x,type=Bool)           v:VariableRef(name=x,type=Bool)
                7:Primary(value=True)                 12:Composite()
            8:Primary(value=True)                         13:Primary(value=True)
            9:Primary(sign=not,ref=x)                     14:Assignment()
                v:VariableRef(name=x,type=Bool)               v:VariableRef(name=x,type=Bool)
            10:Composite()                                    15:Primary(value=True)
                11:Primary(value=True)
            12:Composite()
                13:Primary(value=True)
                14:Assignment()
                    v:VariableRef(name=x,type=Bool)
                    15:Primary(value=True)
        """
        v: Variable = Variable("x")
        v.type = Type("Boolean", 0)

        e1: Transition = Transition(0)
        e2: Primary = Primary(value=True)

        e3: Expression = Expression("or")
        e4: Primary = Primary(value=True)
        e5: Primary = Primary(value=False)
        e3.values = [e4, e5]

        vr: VariableRef = VariableRef()
        vr.var = v

        e6: Assignment = Assignment()
        e7: Primary = Primary(value=True)
        e6.left = vr
        e6.right = e7

        e8: Primary = Primary(value=True)

        vr: VariableRef = VariableRef()
        vr.var = v

        e9: Primary = Primary(value=None)
        e9.ref = vr

        e10: Composite = Composite()
        e11: Primary = Primary(value=True)
        e10.guard = e11

        vr: VariableRef = VariableRef()
        vr.var = v

        e12: Composite = Composite()
        e13: Primary = Primary(value=True)
        e14: Assignment = Assignment()
        e15: Primary = Primary(value=True)
        e14.left = vr
        e14.right = e15
        e12.guard = e13
        e12.assignments = [e14]

        e1.statements = [e2, e3, e6, e8, e9, e10, e12]

        e1.preprocess()

        self.assertEqual(len(e1.statements), 4)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertTrue(e1.statements[0].value)

        self.assertTrue(isinstance(e1.statements[1], Assignment))
        self.assertIs(e1.statements[1], e6)

        self.assertTrue(isinstance(e1.statements[2], Primary))
        self.assertIs(e1.statements[2], e9)

        self.assertTrue(isinstance(e1.statements[3], Composite))
        self.assertIs(e1.statements[3], e12)

        validate_parent_child_relationship(self, e1)

    def test_simplify_transition_true_expression_insertion(self):
        """
        Test the simplify method of the transition class, with the transition starting with an Assignment.
        1:Transition()                    =>      1:Transition()
            2:Assignment()                            4:Primary(value=True)
                v:VariableRef(name=x,type=Bool)       2:Assignment(value=False)
                3:Primary(value=True)                     v:VariableRef(name=x,type=Bool)
                                                          3:Primary(value=True)
        """
        e1: Transition = Transition(0)

        v: Variable = Variable("x")
        v.type = Type("Boolean", 0)
        vr: VariableRef = VariableRef()
        vr.var = v

        e2: Assignment = Assignment()
        e3: Primary = Primary(value=True)
        e2.left = vr
        e2.right = e3
        e1.statements = [e2]

        e1.preprocess()

        self.assertEqual(len(e1.statements), 2)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertTrue(e1.statements[0].value)

        self.assertTrue(isinstance(e1.statements[1], Assignment))
        self.assertIs(e1.statements[1], e2)

        validate_parent_child_relationship(self, e1)

    def test_simplify_transition_false_cutoff(self):
        """
        Test the simplify method of the transition class, with a false statement that prematurely ends the transition.
        1:Transition()                    =>      1:Transition()
            2:Primary(sign=not,ref=x)                     2:Primary(sign=not,ref=x)
                v:VariableRef(name=x,type=Bool)               v:VariableRef(name=x,type=Bool)
            3:Primary(sign=not,value=True)                4:Primary(value=False)
        """
        e1: Transition = Transition(0)

        v: Variable = Variable("x")
        v.type = Type("Boolean", 0)
        vr: VariableRef = VariableRef()
        vr.var = v

        e2: Primary = Primary(sign="not", ref=vr)

        e3: Primary = Primary(sign="not", value=True)
        e1.statements = [e2, e3]

        e1.preprocess()

        self.assertEqual(len(e1.statements), 2)
        self.assertTrue(isinstance(e1.statements[0], Primary))
        self.assertIs(e1.statements[0], e2)

        self.assertTrue(isinstance(e1.statements[1], Primary))
        self.assertFalse(e1.statements[1].value)

        validate_parent_child_relationship(self, e1)


# noinspection DuplicatedCode
class SimplifiableSimplifyComposite(TestCase):
    def test_simplify_composite(self):
        pass


# noinspection DuplicatedCode
class SimplifiableSimplifyExpression(TestCase):
    def test_simplify_composite(self):
        pass
