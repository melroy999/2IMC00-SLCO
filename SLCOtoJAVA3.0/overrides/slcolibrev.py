from os.path import join

import libraries.slcolib


# Overwrite the expression statements since they are right-associative instead of left-associative.
class Expression(object):
    def __init__(self, parent, left=None, op=None, right=None, terms=None):
        self.parent = parent
        if terms is None:
            self.left = left
            self.op = op
            self.right = right
        else:
            if len(terms) > 3:
                self.left = Expression(self, None, None, None, terms[:-2])
                self.op = terms[-2]
                self.right = terms[-1]
            elif len(terms) > 1:
                self.left = terms[0]
                self.op = terms[1]
                self.right = terms[2]
            else:
                self.left = terms[0]
                self.op = ""
                self.right = None


class ExprPrec1(object):
    def __init__(self, parent, left, op, right):
        self.parent = parent
        self.left = left
        self.op = op
        self.right = right


class ExprPrec2(object):
    def __init__(self, parent, left=None, op=None, right=None, terms=None):
        self.parent = parent
        if terms is None:
            self.left = left
            self.op = op
            self.right = right
        else:
            if len(terms) > 3:
                self.left = ExprPrec2(self, None, None, None, terms[:-2])
                self.op = terms[-2]
                self.right = terms[-1]
            elif len(terms) > 1:
                self.left = terms[0]
                self.op = terms[1]
                self.right = terms[2]
            else:
                self.left = terms[0]
                self.op = ""
                self.right = None


class ExprPrec3(object):
    def __init__(self, parent, left=None, op=None, right=None, terms=None):
        self.parent = parent
        if terms is None:
            self.left = left
            self.op = op
            self.right = right
        else:
            if len(terms) > 3:
                self.left = ExprPrec3(self, None, None, None, terms[:-2])
                self.op = terms[-2]
                self.right = terms[-1]
            elif len(terms) > 1:
                self.left = terms[0]
                self.op = terms[1]
                self.right = terms[2]
            else:
                self.left = terms[0]
                self.op = ""
                self.right = None


class ExprPrec4(object):
    def __init__(self, parent, left=None, op=None, right=None, terms=None):
        self.parent = parent
        if terms is None:
            self.left = left
            self.op = op
            self.right = right
        else:
            if len(terms) > 3:
                self.left = ExprPrec4(self, None, None, None, terms[:-2])
                self.op = terms[-2]
                self.right = terms[-1]
            elif len(terms) > 1:
                self.left = terms[0]
                self.op = terms[1]
                self.right = terms[2]
            else:
                self.left = terms[0]
                self.op = ""
                self.right = None


# Overwrite the functions.
libraries.slcolib.Expression = Expression
libraries.slcolib.ExprPrec1 = ExprPrec1
libraries.slcolib.ExprPrec2 = ExprPrec2
libraries.slcolib.ExprPrec3 = ExprPrec3
libraries.slcolib.ExprPrec4 = ExprPrec4


# Next, overwrite the main function to use the correct grammar.
# noinspection PyPep8Naming
def read_SLCO_model(m):
    """Read, post process, and type check an SLCO model"""
    # create meta-model
    slco_mm = libraries.slcolib.metamodel_from_file(
        join(libraries.slcolib.this_folder, '../textx_grammars/slco2_left_associative.tx'),
        autokwd=True,
        classes=[
            libraries.slcolib.Assignment,
            libraries.slcolib.Composite,
            Expression,
            ExprPrec1,
            ExprPrec2,
            ExprPrec3,
            ExprPrec4,
            libraries.slcolib.Primary,
            libraries.slcolib.ExpressionRef,
            libraries.slcolib.Variable,
            libraries.slcolib.VariableRef,
            libraries.slcolib.Type,
            libraries.slcolib.Action
        ]
    )

    # register processors
    slco_mm.register_model_processor(libraries.slcolib.construct_action_set)
    slco_mm.register_model_processor(libraries.slcolib.check_names)
    slco_mm.register_model_processor(libraries.slcolib.add_initial_to_states)
    slco_mm.register_model_processor(libraries.slcolib.add_variable_types)
    # slco_mm.register_model_processor(libraries.slcolib.set_default_type_size)
    slco_mm.register_model_processor(libraries.slcolib.set_default_channel_size)
    slco_mm.register_model_processor(libraries.slcolib.add_taus)
    slco_mm.register_model_processor(libraries.slcolib.fix_references)
    # slco_mm.register_model_processor(libraries.slcolib.simplify_statements)

    # To do: Check receive statements for not receiving multiple values in the same variable
    # To do: Check for absence of arrays (not single elements) as part of messages

    slco_mm.register_scope_providers({
        "*.*": libraries.slcolib.providers.FQN(),
        "Initialisation.left": libraries.slcolib.providers.RelativeName("parent.type.variables"),
        "Channel.port0": libraries.slcolib.providers.RelativeName("source.type.ports"),
        "Channel.port1": libraries.slcolib.providers.RelativeName("target.type.ports"),
        "ReceiveSignal.from": libraries.slcolib.providers.RelativeName("parent.parent.parent.type.ports"),
        "SendSignal.to": libraries.slcolib.providers.RelativeName("parent.parent.parent.type.ports"),
    })

    # parse and return the model
    return slco_mm.model_from_file(m)
