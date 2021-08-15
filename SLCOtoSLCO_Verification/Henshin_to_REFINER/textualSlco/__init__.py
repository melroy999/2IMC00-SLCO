from .textualSlco import getEClassifier, eClassifiers
from .textualSlco import name, nsURI, nsPrefix, eClass
from .textualSlco import Model, Class, StateMachine, Vertex, Initial, State, Final, Transition, Statement, Delay, Action, DoAction, SignalReception, SignalArgument, SignalArgumentVariable, SignalArgumentExpression, Assignment, SendSignal, Object, Port, Channel, UnidirectionalChannel, BidirectionalChannel, ArgumentType, Variable, ConstantArray, VariableType, PrimitiveType, ChannelType, Composite, Expression, Operator, BooleanConstantExpression, IntegerConstantExpression, VariableReference, VariableExpression, BinaryOperatorExpression
from . import textualSlco
from pyecore.ecore import EObject

__all__ = ['Model', 'Class', 'StateMachine', 'Vertex', 'Initial', 'State', 'Final', 'Transition', 'Statement', 'Delay', 'Action', 'DoAction', 'SignalReception', 'SignalArgument', 'SignalArgumentVariable', 'SignalArgumentExpression', 'Assignment', 'SendSignal', 'Object', 'Port', 'Channel', 'UnidirectionalChannel', 'BidirectionalChannel', 'ArgumentType', 'Variable', 'ConstantArray', 'VariableType', 'PrimitiveType', 'ChannelType', 'Composite', 'Expression', 'Operator', 'BooleanConstantExpression', 'IntegerConstantExpression', 'VariableReference', 'VariableExpression', 'BinaryOperatorExpression']

eSubpackages = []
eSuperPackage = None

# Non opposite EReferences
Model.actions.eType = Action
Model.classes.eType = Class
Model.objects.eType = Object
Model.channels.eType = Channel
Class.variables.eType = Variable
Class.ports.eType = Port
Class.stateMachines.eType = StateMachine
StateMachine.variables.eType = Variable
StateMachine.vertices.eType = Vertex
StateMachine.transitions.eType = Transition
Transition.source.eType = Vertex
Transition.target.eType = Vertex
Transition.statements.eType = Statement
SignalReception.arguments.eType = SignalArgument
SignalReception.condition.eType = Expression
SignalReception.port.eType = Port
SignalArgumentVariable.variable.eType = Variable
SignalArgumentExpression.expression.eType = Expression
Assignment.variable.eType = VariableReference
Assignment.expression.eType = Expression
SendSignal.params.eType = Expression
SendSignal.target.eType = Port
Object.oclass.eType = Class
Object.assignments.eType = Assignment
Channel.argumentTypes.eType = ArgumentType
Channel.channelType.eType = ChannelType
Channel.source.eType = Object
Channel.target.eType = Object
UnidirectionalChannel.sourcePort.eType = Port
UnidirectionalChannel.targetPort.eType = Port
BidirectionalChannel.port1.eType = Port
BidirectionalChannel.port2.eType = Port
ArgumentType.type.eType = VariableType
Variable.type.eType = VariableType
Variable.initialValue.eType = EObject
ConstantArray.values.eType = Expression
Composite.guard.eType = Expression
Composite.assignments.eType = Assignment
VariableReference.var.eType = Variable
VariableReference.index.eType = Expression
VariableExpression.variable.eType = VariableReference
BinaryOperatorExpression.operand1.eType = Expression
BinaryOperatorExpression.operand2.eType = Expression


# Manage all other EClassifiers (EEnum, EDatatypes...)
otherClassifiers = [PrimitiveType, Operator]
for classif in otherClassifiers:
    eClassifiers[classif.name] = classif
    classif._container = textualSlco

for classif in eClassifiers.values():
    eClass.eClassifiers.append(classif.eClass)

for subpack in eSubpackages:
    eClass.eSubpackages.append(subpack.eClass)
