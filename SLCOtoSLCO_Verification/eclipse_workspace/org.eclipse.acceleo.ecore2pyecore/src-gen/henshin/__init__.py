from .henshin import getEClassifier, eClassifiers
from .henshin import name, nsURI, nsPrefix, eClass
from .henshin import NamedElement, GraphElement, Module, Unit, Rule, Parameter, ParameterMapping, Graph, Node, Edge, Attribute, AttributeCondition, Mapping, UnaryUnit, MultiUnit, IndependentUnit, SequentialUnit, ConditionalUnit, PriorityUnit, IteratedUnit, LoopUnit, Formula, NestedCondition, UnaryFormula, BinaryFormula, And, Or, Xor, Not, Action
from . import henshin

__all__ = ['NamedElement', 'GraphElement', 'Module', 'Unit', 'Rule', 'Parameter', 'ParameterMapping', 'Graph', 'Node', 'Edge', 'Attribute', 'AttributeCondition', 'Mapping', 'UnaryUnit', 'MultiUnit', 'IndependentUnit', 'SequentialUnit', 'ConditionalUnit', 'PriorityUnit', 'IteratedUnit', 'LoopUnit', 'Formula', 'NestedCondition', 'UnaryFormula', 'BinaryFormula', 'And', 'Or', 'Xor', 'Not', 'Action']

eSubpackages = []
eSuperPackage = None

# Non opposite EReferences
Module.imports.eType = EPackage
Module.units.eType = Unit
Module.instances.eType = Graph
Unit.parameterMappings.eType = ParameterMapping
Rule.lhs.eType = Graph
Rule.rhs.eType = Graph
Rule.mappings.eType = Mapping
Rule.multiRules.eType = Rule
Rule.multiMappings.eType = Mapping
Parameter.type.eType = EClassifier
ParameterMapping.source.eType = Parameter
ParameterMapping.target.eType = Parameter
Graph.formula.eType = Formula
Node.type.eType = EClass
Edge.type.eType = EReference
Attribute.type.eType = EAttribute
Mapping.origin.eType = Node
Mapping.image.eType = Node
UnaryUnit.subUnit.eType = Unit
MultiUnit.subUnits.eType = Unit
ConditionalUnit.if.eType = Unit
ConditionalUnit.then.eType = Unit
ConditionalUnit.else.eType = Unit
NestedCondition.conclusion.eType = Graph
NestedCondition.mappings.eType = Mapping
UnaryFormula.child.eType = Formula
BinaryFormula.left.eType = Formula
BinaryFormula.right.eType = Formula

# opposite EReferences
Module.subModules.eType = Module
Module.superModule.eType = Module
Module.superModule.eOpposite = Module.subModules
Unit.parameters.eType = Parameter
Rule.attributeConditions.eType = AttributeCondition
Parameter.unit.eType = Unit
Parameter.unit.eOpposite = Unit.parameters
Graph.nodes.eType = Node
Graph.edges.eType = Edge
Node.attributes.eType = Attribute
Node.graph.eType = Graph
Node.graph.eOpposite = Graph.nodes
Node.incoming.eType = Edge
Node.outgoing.eType = Edge
Edge.source.eType = Node
Edge.source.eOpposite = Node.outgoing
Edge.target.eType = Node
Edge.target.eOpposite = Node.incoming
Edge.graph.eType = Graph
Edge.graph.eOpposite = Graph.edges
Attribute.node.eType = Node
Attribute.node.eOpposite = Node.attributes
AttributeCondition.rule.eType = Rule
AttributeCondition.rule.eOpposite = Rule.attributeConditions


# Manage all other EClassifiers (EEnum, EDatatypes...)
otherClassifiers = [Action]
for classif in otherClassifiers:
    eClassifiers[classif.name] = classif
    classif._container = henshin

for classif in eClassifiers.values():
    eClass.eClassifiers.append(classif.eClass)

for subpack in eSubpackages:
    eClass.eSubpackages.append(subpack.eClass)
