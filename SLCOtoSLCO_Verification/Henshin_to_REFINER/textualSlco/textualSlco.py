from functools import partial
import pyecore.ecore as Ecore
from pyecore.ecore import *

name = 'textualSlco'
nsURI = 'http://www.xtext.org/slco/textualslco/TextualSlco'
nsPrefix = 'textualSlco'

eClass = EPackage(name=name, nsURI=nsURI, nsPrefix=nsPrefix)

eClassifiers = {}
getEClassifier = partial(Ecore.getEClassifier, searchspace=eClassifiers)


PrimitiveType = EEnum('PrimitiveType', literals=['Integer', 'Boolean'])  # noqa
Operator = EEnum('Operator', literals=['moreThan', 'lessThan', 'atLeast', 'atMost', 'and', 'or', 'equals', 'differs', 'mutiplication', 'add', 'subtract', 'modulo'])  # noqa


class Model(EObject, metaclass=MetaEClass):
    name = EAttribute(eType=EString)
    actions = EReference(upper=-1, containment=True)
    classes = EReference(upper=-1, containment=True)
    objects = EReference(upper=-1, containment=True)
    channels = EReference(upper=-1, containment=True)

    def __init__(self, actions=None, name=None, classes=None, objects=None, channels=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if name is not None:
            self.name = name
        if actions:
            self.actions.extend(actions)
        if classes:
            self.classes.extend(classes)
        if objects:
            self.objects.extend(objects)
        if channels:
            self.channels.extend(channels)


class Class(EObject, metaclass=MetaEClass):
    name = EAttribute(eType=EString)
    variables = EReference(upper=-1, containment=True)
    ports = EReference(upper=-1, containment=True)
    stateMachines = EReference(upper=-1, containment=True)

    def __init__(self, name=None, variables=None, ports=None, stateMachines=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if name is not None:
            self.name = name
        if variables:
            self.variables.extend(variables)
        if ports:
            self.ports.extend(ports)
        if stateMachines:
            self.stateMachines.extend(stateMachines)


class StateMachine(EObject, metaclass=MetaEClass):
    name = EAttribute(eType=EString)
    variables = EReference(upper=-1, containment=True)
    vertices = EReference(upper=-1, containment=True)
    transitions = EReference(upper=-1, containment=True)

    def __init__(self, name=None, variables=None, vertices=None, transitions=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if name is not None:
            self.name = name
        if variables:
            self.variables.extend(variables)
        if vertices:
            self.vertices.extend(vertices)
        if transitions:
            self.transitions.extend(transitions)


class Vertex(EObject, metaclass=MetaEClass):
    name = EAttribute(eType=EString)

    def __init__(self, name=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if name is not None:
            self.name = name


class Transition(EObject, metaclass=MetaEClass):
    priority = EAttribute(eType=EInt)
    source = EReference()
    target = EReference()
    statements = EReference(upper=-1, containment=True)

    def __init__(self, priority=None, source=None, target=None, statements=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if priority is not None:
            self.priority = priority
        if source is not None:
            self.source = source
        if target is not None:
            self.target = target
        if statements:
            self.statements.extend(statements)


class Statement(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()


class Action(EObject, metaclass=MetaEClass):
    name = EAttribute(eType=EString)

    def __init__(self, name=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if name is not None:
            self.name = name


class SignalArgument(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()


class Object(EObject, metaclass=MetaEClass):
    name = EAttribute(eType=EString)
    oclass = EReference()
    assignments = EReference(upper=-1, containment=True)

    def __init__(self, name=None, oclass=None, assignments=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if name is not None:
            self.name = name
        if oclass is not None:
            self.oclass = oclass
        if assignments:
            self.assignments.extend(assignments)


class Port(EObject, metaclass=MetaEClass):
    name = EAttribute(eType=EString)

    def __init__(self, name=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if name is not None:
            self.name = name


class Channel(EObject, metaclass=MetaEClass):
    name = EAttribute(eType=EString)
    argumentTypes = EReference(upper=-1, containment=True)
    channelType = EReference(containment=True)
    source = EReference()
    target = EReference()

    def __init__(self, name=None, argumentTypes=None, channelType=None, source=None, target=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if name is not None:
            self.name = name
        if argumentTypes:
            self.argumentTypes.extend(argumentTypes)
        if channelType is not None:
            self.channelType = channelType
        if source is not None:
            self.source = source
        if target is not None:
            self.target = target


class ArgumentType(EObject, metaclass=MetaEClass):
    type = EReference(containment=True)

    def __init__(self, type=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if type is not None:
            self.type = type


class Variable(EObject, metaclass=MetaEClass):
    name = EAttribute(eType=EString)
    type = EReference(containment=True)
    initialValue = EReference(containment=True)

    def __init__(self, type=None, name=None, initialValue=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if name is not None:
            self.name = name
        if type is not None:
            self.type = type
        if initialValue is not None:
            self.initialValue = initialValue


class ConstantArray(EObject, metaclass=MetaEClass):
    values = EReference(upper=-1, containment=True)

    def __init__(self, values=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if values:
            self.values.extend(values)


class VariableType(EObject, metaclass=MetaEClass):
    base = EAttribute(eType=PrimitiveType)
    size = EAttribute(eType=EInt)

    def __init__(self, base=None, size=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if base is not None:
            self.base = base
        if size is not None:
            self.size = size


class ChannelType(EObject, metaclass=MetaEClass):
    Synchronous = EAttribute(eType=EBoolean)
    Asynchronous = EAttribute(eType=EBoolean)
    size = EAttribute(eType=EInt)
    losstype = EAttribute(eType=EString)

    def __init__(self, Synchronous=None, Asynchronous=None, size=None, losstype=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if Synchronous is not None:
            self.Synchronous = Synchronous
        if Asynchronous is not None:
            self.Asynchronous = Asynchronous
        if size is not None:
            self.size = size
        if losstype is not None:
            self.losstype = losstype


class VariableReference(EObject, metaclass=MetaEClass):
    var = EReference()
    index = EReference(containment=True)

    def __init__(self, var=None, index=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if var is not None:
            self.var = var
        if index is not None:
            self.index = index


class Initial(Vertex):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class State(Vertex):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Final(Vertex):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Delay(Statement):
    length = EAttribute(eType=EInt)

    def __init__(self, length=None, **kwargs):
        super().__init__(**kwargs)
        if length is not None:
            self.length = length


class DoAction(Statement):
    name = EAttribute(eType=EString)

    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        if name is not None:
            self.name = name


class SignalReception(Statement):
    signalName = EAttribute(eType=EString)
    arguments = EReference(upper=-1, containment=True)
    condition = EReference(containment=True)
    port = EReference()

    def __init__(self, signalName=None, arguments=None, condition=None, port=None, **kwargs):
        super().__init__(**kwargs)
        if signalName is not None:
            self.signalName = signalName
        if arguments:
            self.arguments.extend(arguments)
        if condition is not None:
            self.condition = condition
        if port is not None:
            self.port = port


class SignalArgumentVariable(SignalArgument):
    variable = EReference()

    def __init__(self, variable=None, **kwargs):
        super().__init__(**kwargs)
        if variable is not None:
            self.variable = variable


class SignalArgumentExpression(SignalArgument):
    expression = EReference(containment=True)

    def __init__(self, expression=None, **kwargs):
        super().__init__(**kwargs)
        if expression is not None:
            self.expression = expression


class Assignment(Statement):
    variable = EReference(containment=True)
    expression = EReference(containment=True)

    def __init__(self, variable=None, expression=None, **kwargs):
        super().__init__(**kwargs)
        if variable is not None:
            self.variable = variable
        if expression is not None:
            self.expression = expression


class SendSignal(Statement):
    signal = EAttribute(eType=EString)
    params = EReference(upper=-1, containment=True)
    target = EReference()

    def __init__(self, signal=None, params=None, target=None, **kwargs):
        super().__init__(**kwargs)
        if signal is not None:
            self.signal = signal
        if params:
            self.params.extend(params)
        if target is not None:
            self.target = target


class UnidirectionalChannel(Channel):
    sourcePort = EReference()
    targetPort = EReference()

    def __init__(self, sourcePort=None, targetPort=None, **kwargs):
        super().__init__(**kwargs)
        if sourcePort is not None:
            self.sourcePort = sourcePort
        if targetPort is not None:
            self.targetPort = targetPort


class BidirectionalChannel(Channel):
    port1 = EReference()
    port2 = EReference()

    def __init__(self, port1=None, port2=None, **kwargs):
        super().__init__(**kwargs)
        if port1 is not None:
            self.port1 = port1
        if port2 is not None:
            self.port2 = port2


class Composite(Statement):
    guard = EReference(containment=True)
    assignments = EReference(upper=-1, containment=True)

    def __init__(self, guard=None, assignments=None, **kwargs):
        super().__init__(**kwargs)
        if guard is not None:
            self.guard = guard
        if assignments:
            self.assignments.extend(assignments)


class Expression(Statement):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BooleanConstantExpression(Expression):
    value = EAttribute(eType=EString)

    def __init__(self, value=None, **kwargs):
        super().__init__(**kwargs)
        if value is not None:
            self.value = value


class IntegerConstantExpression(Expression):
    value = EAttribute(eType=EInt)

    def __init__(self, value=None, **kwargs):
        super().__init__(**kwargs)
        if value is not None:
            self.value = value


class VariableExpression(Expression):
    variable = EReference(containment=True)

    def __init__(self, variable=None, **kwargs):
        super().__init__(**kwargs)
        if variable is not None:
            self.variable = variable


class BinaryOperatorExpression(Expression):
    operator = EAttribute(eType=Operator)
    operand1 = EReference(containment=True)
    operand2 = EReference(containment=True)

    def __init__(self, operand1=None, operator=None, operand2=None, **kwargs):
        super().__init__(**kwargs)
        if operator is not None:
            self.operator = operator
        if operand1 is not None:
            self.operand1 = operand1
        if operand2 is not None:
            self.operand2 = operand2
