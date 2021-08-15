from functools import partial
import pyecore.ecore as Ecore
from pyecore.ecore import *

name = 'henshin'
nsURI = 'http://www.eclipse.org/emf/2011/Henshin'
nsPrefix = 'henshin'

eClass = EPackage(name=name, nsURI=nsURI, nsPrefix=nsPrefix)

eClassifiers = {}
getEClassifier = partial(Ecore.getEClassifier, searchspace=eClassifiers)


Action = EDataType('Action', instanceClassName='org.eclipse.emf.henshin.model.Action')


@abstract
class NamedElement(EObject, metaclass=MetaEClass):
    name = EAttribute(eType=EString)
    description = EAttribute(eType=EString)

    def __init__(self, name=None, description=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description


@abstract
class GraphElement(EObject, metaclass=MetaEClass):
    action = EAttribute(eType=Action)

    def __init__(self, action=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if action is not None:
            self.action = action
    def getGraph(self):
        raise NotImplementedError('Operation getGraph(...) is not yet implemented')


class ParameterMapping(EObject, metaclass=MetaEClass):
    source = EReference()
    target = EReference()

    def __init__(self, source=None, target=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if source is not None:
            self.source = source
        if target is not None:
            self.target = target


class Mapping(EObject, metaclass=MetaEClass):
    origin = EReference()
    image = EReference()

    def __init__(self, origin=None, image=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
        if origin is not None:
            self.origin = origin
        if image is not None:
            self.image = image


@abstract
class Formula(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()
    def isTrue(self):
        raise NotImplementedError('Operation isTrue(...) is not yet implemented')
    def isFalse(self):
        raise NotImplementedError('Operation isFalse(...) is not yet implemented')


class Module(NamedElement):
    subModules = EReference(upper=-1, containment=True)
    superModule = EReference()
    imports = EReference(upper=-1)
    units = EReference(upper=-1, containment=True)
    instances = EReference(upper=-1, containment=True)

    def __init__(self, subModules=None, superModule=None, imports=None, units=None, instances=None, **kwargs):
        super().__init__(**kwargs)
        if subModules:
            self.subModules.extend(subModules)
        if superModule is not None:
            self.superModule = superModule
        if imports:
            self.imports.extend(imports)
        if units:
            self.units.extend(units)
        if instances:
            self.instances.extend(instances)
    def getUnit(self, name=None):
        raise NotImplementedError('Operation getUnit(...) is not yet implemented')
    def getSubModule(self, name=None):
        raise NotImplementedError('Operation getSubModule(...) is not yet implemented')


@abstract
class Unit(NamedElement):
    activated = EAttribute(eType=EBoolean)
    parameters = EReference(upper=-1, containment=True)
    parameterMappings = EReference(upper=-1, containment=True)

    def __init__(self, parameters=None, parameterMappings=None, activated=None, **kwargs):
        super().__init__(**kwargs)
        if activated is not None:
            self.activated = activated
        if parameters:
            self.parameters.extend(parameters)
        if parameterMappings:
            self.parameterMappings.extend(parameterMappings)
    def getModule(self):
        raise NotImplementedError('Operation getModule(...) is not yet implemented')
    def getSubUnits(self, deep=None):
        raise NotImplementedError('Operation getSubUnits(...) is not yet implemented')
    def getParameter(self, parameter=None):
        raise NotImplementedError('Operation getParameter(...) is not yet implemented')


class Parameter(NamedElement):
    unit = EReference()
    type = EReference()

    def __init__(self, unit=None, type=None, **kwargs):
        super().__init__(**kwargs)
        if unit is not None:
            self.unit = unit
        if type is not None:
            self.type = type


class Graph(NamedElement):
    nodes = EReference(upper=-1, containment=True)
    edges = EReference(upper=-1, containment=True)
    formula = EReference(containment=True)

    def __init__(self, nodes=None, edges=None, formula=None, **kwargs):
        super().__init__(**kwargs)
        if nodes:
            self.nodes.extend(nodes)
        if edges:
            self.edges.extend(edges)
        if formula is not None:
            self.formula = formula
    def getRule(self):
        raise NotImplementedError('Operation getRule(...) is not yet implemented')
    def getNode(self, name=None):
        raise NotImplementedError('Operation getNode(...) is not yet implemented')
    def getNodes(self, nodeType):
        raise NotImplementedError('Operation getNodes(...) is not yet implemented')
    def getEdges(self, edgeType):
        raise NotImplementedError('Operation getEdges(...) is not yet implemented')
    def getNestedConditions(self):
        raise NotImplementedError('Operation getNestedConditions(...) is not yet implemented')
    def getPAC(self, name=None):
        raise NotImplementedError('Operation getPAC(...) is not yet implemented')
    def getNAC(self, name=None):
        raise NotImplementedError('Operation getNAC(...) is not yet implemented')
    def getPACs(self):
        raise NotImplementedError('Operation getPACs(...) is not yet implemented')
    def getNACs(self):
        raise NotImplementedError('Operation getNACs(...) is not yet implemented')
    def isLhs(self):
        raise NotImplementedError('Operation isLhs(...) is not yet implemented')
    def isRhs(self):
        raise NotImplementedError('Operation isRhs(...) is not yet implemented')
    def isNestedCondition(self):
        raise NotImplementedError('Operation isNestedCondition(...) is not yet implemented')
    def createPAC(self, name=None):
        raise NotImplementedError('Operation createPAC(...) is not yet implemented')
    def createNAC(self, name=None):
        raise NotImplementedError('Operation createNAC(...) is not yet implemented')
    def removeNode(self, node=None):
        raise NotImplementedError('Operation removeNode(...) is not yet implemented')
    def removeEdge(self, edge):
        raise NotImplementedError('Operation removeEdge(...) is not yet implemented')
    def removeNestedCondition(self, nestedCondition=None):
        raise NotImplementedError('Operation removeNestedCondition(...) is not yet implemented')


class Edge(GraphElement):
    index = EAttribute(eType=EString)
    _indexConstant = EAttribute(name='indexConstant', eType=EIntegerObject, derived=True, changeable=False)
    source = EReference()
    target = EReference()
    type = EReference()
    graph = EReference()

    @property
    def indexConstant(self):
        return self._indexConstant

    @indexConstant.setter
    def indexConstant(self, value):
        self._indexConstant = value

    def __init__(self, source=None, target=None, type=None, graph=None, index=None, indexConstant=None, **kwargs):
        super().__init__(**kwargs)
        if index is not None:
            self.index = index
        if indexConstant is not None:
            self.indexConstant = indexConstant
        if source is not None:
            self.source = source
        if target is not None:
            self.target = target
        if type is not None:
            self.type = type
        if graph is not None:
            self.graph = graph
    def getActionEdge(self):
        raise NotImplementedError('Operation getActionEdge(...) is not yet implemented')


class Attribute(GraphElement):
    value = EAttribute(eType=EString)
    _constant = EAttribute(name='constant', eType=EJavaObject, derived=True, changeable=False)
    _null = EAttribute(name='null', eType=EBoolean, derived=True, changeable=False)
    type = EReference()
    node = EReference()

    @property
    def constant(self):
        return self._constant

    @constant.setter
    def constant(self, value):
        self._constant = value

    @property
    def null(self):
        return self._null

    @null.setter
    def null(self, value):
        self._null = value

    def __init__(self, type=None, value=None, node=None, constant=None, null=None, **kwargs):
        super().__init__(**kwargs)
        if value is not None:
            self.value = value
        if constant is not None:
            self.constant = constant
        if null is not None:
            self.null = null
        if type is not None:
            self.type = type
        if node is not None:
            self.node = node
    def getActionAttribute(self):
        raise NotImplementedError('Operation getActionAttribute(...) is not yet implemented')


class AttributeCondition(NamedElement):
    conditionText = EAttribute(eType=EString)
    rule = EReference()

    def __init__(self, rule=None, conditionText=None, **kwargs):
        super().__init__(**kwargs)
        if conditionText is not None:
            self.conditionText = conditionText
        if rule is not None:
            self.rule = rule


class NestedCondition(Formula):
    conclusion = EReference(containment=True)
    mappings = EReference(upper=-1, containment=True)

    def __init__(self, conclusion=None, mappings=None, **kwargs):
        super().__init__(**kwargs)
        if conclusion is not None:
            self.conclusion = conclusion
        if mappings:
            self.mappings.extend(mappings)
    def getHost(self):
        raise NotImplementedError('Operation getHost(...) is not yet implemented')
    def isPAC(self):
        raise NotImplementedError('Operation isPAC(...) is not yet implemented')
    def isNAC(self):
        raise NotImplementedError('Operation isNAC(...) is not yet implemented')


@abstract
class UnaryFormula(Formula):
    child = EReference(containment=True)

    def __init__(self, child=None, **kwargs):
        super().__init__(**kwargs)
        if child is not None:
            self.child = child


@abstract
class BinaryFormula(Formula):
    left = EReference(containment=True)
    right = EReference(containment=True)

    def __init__(self, left=None, right=None, **kwargs):
        super().__init__(**kwargs)
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right


class Rule(Unit):
    checkDangling = EAttribute(eType=EBoolean)
    injectiveMatching = EAttribute(eType=EBoolean)
    javaImports = EAttribute(eType=EString, upper=-1)
    lhs = EReference(containment=True)
    rhs = EReference(containment=True)
    attributeConditions = EReference(upper=-1, containment=True)
    mappings = EReference(upper=-1, containment=True)
    multiRules = EReference(upper=-1, containment=True)
    multiMappings = EReference(upper=-1, containment=True)

    def __init__(self, lhs=None, rhs=None, attributeConditions=None, mappings=None, checkDangling=None, injectiveMatching=None, multiRules=None, multiMappings=None, javaImports=None, **kwargs):
        super().__init__(**kwargs)
        if checkDangling is not None:
            self.checkDangling = checkDangling
        if injectiveMatching is not None:
            self.injectiveMatching = injectiveMatching
        if javaImports:
            self.javaImports.extend(javaImports)
        if lhs is not None:
            self.lhs = lhs
        if rhs is not None:
            self.rhs = rhs
        if attributeConditions:
            self.attributeConditions.extend(attributeConditions)
        if mappings:
            self.mappings.extend(mappings)
        if multiRules:
            self.multiRules.extend(multiRules)
        if multiMappings:
            self.multiMappings.extend(multiMappings)
    def getKernelRule(self):
        raise NotImplementedError('Operation getKernelRule(...) is not yet implemented')
    def getRootRule(self):
        raise NotImplementedError('Operation getRootRule(...) is not yet implemented')
    def getMultiRule(self, name=None):
        raise NotImplementedError('Operation getMultiRule(...) is not yet implemented')
    def getMultiRulePath(self, multiRule=None):
        raise NotImplementedError('Operation getMultiRulePath(...) is not yet implemented')
    def getAllMultiRules(self):
        raise NotImplementedError('Operation getAllMultiRules(...) is not yet implemented')
    def getAllMappings(self):
        raise NotImplementedError('Operation getAllMappings(...) is not yet implemented')
    def getActionNodes(self, action=None):
        raise NotImplementedError('Operation getActionNodes(...) is not yet implemented')
    def getActionEdges(self, action=None):
        raise NotImplementedError('Operation getActionEdges(...) is not yet implemented')
    def getParameterNodes(self):
        raise NotImplementedError('Operation getParameterNodes(...) is not yet implemented')
    def isMultiRule(self):
        raise NotImplementedError('Operation isMultiRule(...) is not yet implemented')
    def createNode(self, type=None):
        raise NotImplementedError('Operation createNode(...) is not yet implemented')
    def createEdge(self, source=None, target=None, type=None):
        raise NotImplementedError('Operation createEdge(...) is not yet implemented')
    def canCreateEdge(self, source=None, target=None, type=None):
        raise NotImplementedError('Operation canCreateEdge(...) is not yet implemented')
    def removeEdge(self, edge=None, removeMapped=None):
        raise NotImplementedError('Operation removeEdge(...) is not yet implemented')
    def removeNode(self, node=None, removeMapped=None):
        raise NotImplementedError('Operation removeNode(...) is not yet implemented')
    def removeAttribute(self, attribute=None, removeMapped=None):
        raise NotImplementedError('Operation removeAttribute(...) is not yet implemented')


class Node(NamedElement, GraphElement):
    type = EReference()
    attributes = EReference(upper=-1, containment=True)
    graph = EReference()
    incoming = EReference(upper=-1)
    outgoing = EReference(upper=-1)

    def __init__(self, type=None, attributes=None, graph=None, incoming=None, outgoing=None, **kwargs):
        super().__init__(**kwargs)
        if type is not None:
            self.type = type
        if attributes:
            self.attributes.extend(attributes)
        if graph is not None:
            self.graph = graph
        if incoming:
            self.incoming.extend(incoming)
        if outgoing:
            self.outgoing.extend(outgoing)
    def getAllEdges(self):
        raise NotImplementedError('Operation getAllEdges(...) is not yet implemented')
    def getOutgoing(self, type):
        raise NotImplementedError('Operation getOutgoing(...) is not yet implemented')
    def getIncoming(self, type):
        raise NotImplementedError('Operation getIncoming(...) is not yet implemented')
    def getOutgoing(self, type, target):
        raise NotImplementedError('Operation getOutgoing(...) is not yet implemented')
    def getIncoming(self, type, source):
        raise NotImplementedError('Operation getIncoming(...) is not yet implemented')
    def getAttribute(self, type=None):
        raise NotImplementedError('Operation getAttribute(...) is not yet implemented')
    def getActionAttributes(self, action=None):
        raise NotImplementedError('Operation getActionAttributes(...) is not yet implemented')
    def getActionNode(self):
        raise NotImplementedError('Operation getActionNode(...) is not yet implemented')


@abstract
class UnaryUnit(Unit):
    subUnit = EReference()

    def __init__(self, subUnit=None, **kwargs):
        super().__init__(**kwargs)
        if subUnit is not None:
            self.subUnit = subUnit


@abstract
class MultiUnit(Unit):
    subUnits = EReference(upper=-1, unique=False)

    def __init__(self, subUnits=None, **kwargs):
        super().__init__(**kwargs)
        if subUnits:
            self.subUnits.extend(subUnits)


class ConditionalUnit(Unit):
    if = EReference()
    then = EReference()
    else = EReference()

    def __init__(self, if=None, then=None, else=None, **kwargs):
        super().__init__(**kwargs)
        if if is not None:
            self.if = if
        if then is not None:
            self.then = then
        if else is not None:
            self.else = else


class And(BinaryFormula):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Or(BinaryFormula):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Xor(BinaryFormula):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Not(UnaryFormula):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class IndependentUnit(MultiUnit):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SequentialUnit(MultiUnit):
    strict = EAttribute(eType=EBoolean)
    rollback = EAttribute(eType=EBoolean)

    def __init__(self, strict=None, rollback=None, **kwargs):
        super().__init__(**kwargs)
        if strict is not None:
            self.strict = strict
        if rollback is not None:
            self.rollback = rollback


class PriorityUnit(MultiUnit):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class IteratedUnit(UnaryUnit):
    iterations = EAttribute(eType=EString)

    def __init__(self, iterations=None, **kwargs):
        super().__init__(**kwargs)
        if iterations is not None:
            self.iterations = iterations


class LoopUnit(UnaryUnit):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
