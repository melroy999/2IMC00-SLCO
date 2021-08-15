/**
 */
package org.xtext.textualSlco.impl;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EDataType;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.impl.EFactoryImpl;

import org.eclipse.emf.ecore.plugin.EcorePlugin;

import org.xtext.textualSlco.ArgumentType;
import org.xtext.textualSlco.Assignment;
import org.xtext.textualSlco.BidirectionalChannel;
import org.xtext.textualSlco.BinaryOperatorExpression;
import org.xtext.textualSlco.BooleanConstantExpression;
import org.xtext.textualSlco.Channel;
import org.xtext.textualSlco.ChannelType;
import org.xtext.textualSlco.Delay;
import org.xtext.textualSlco.Expression;
import org.xtext.textualSlco.Final;
import org.xtext.textualSlco.Initial;
import org.xtext.textualSlco.IntegerConstantExpression;
import org.xtext.textualSlco.Model;
import org.xtext.textualSlco.Operator;
import org.xtext.textualSlco.Port;
import org.xtext.textualSlco.PortReference;
import org.xtext.textualSlco.PrimitiveType;
import org.xtext.textualSlco.SendSignal;
import org.xtext.textualSlco.SignalArgument;
import org.xtext.textualSlco.SignalArgumentExpression;
import org.xtext.textualSlco.SignalArgumentVariable;
import org.xtext.textualSlco.SignalReception;
import org.xtext.textualSlco.State;
import org.xtext.textualSlco.StateMachine;
import org.xtext.textualSlco.Statement;
import org.xtext.textualSlco.StringConstantExpression;
import org.xtext.textualSlco.TextualSlcoFactory;
import org.xtext.textualSlco.TextualSlcoPackage;
import org.xtext.textualSlco.Transition;
import org.xtext.textualSlco.UnidirectionalChannel;
import org.xtext.textualSlco.Variable;
import org.xtext.textualSlco.VariableExpression;
import org.xtext.textualSlco.VariableReference;
import org.xtext.textualSlco.Vertex;
import org.xtext.textualSlco.VertexReference;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Factory</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class TextualSlcoFactoryImpl extends EFactoryImpl implements TextualSlcoFactory
{
  /**
   * Creates the default factory implementation.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public static TextualSlcoFactory init()
  {
    try
    {
      TextualSlcoFactory theTextualSlcoFactory = (TextualSlcoFactory)EPackage.Registry.INSTANCE.getEFactory(TextualSlcoPackage.eNS_URI);
      if (theTextualSlcoFactory != null)
      {
        return theTextualSlcoFactory;
      }
    }
    catch (Exception exception)
    {
      EcorePlugin.INSTANCE.log(exception);
    }
    return new TextualSlcoFactoryImpl();
  }

  /**
   * Creates an instance of the factory.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public TextualSlcoFactoryImpl()
  {
    super();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public EObject create(EClass eClass)
  {
    switch (eClass.getClassifierID())
    {
      case TextualSlcoPackage.MODEL: return createModel();
      case TextualSlcoPackage.CLASS: return createClass();
      case TextualSlcoPackage.STATE_MACHINE: return createStateMachine();
      case TextualSlcoPackage.VERTEX: return createVertex();
      case TextualSlcoPackage.INITIAL: return createInitial();
      case TextualSlcoPackage.STATE: return createState();
      case TextualSlcoPackage.FINAL: return createFinal();
      case TextualSlcoPackage.VERTEX_REFERENCE: return createVertexReference();
      case TextualSlcoPackage.TRANSITION: return createTransition();
      case TextualSlcoPackage.STATEMENT: return createStatement();
      case TextualSlcoPackage.DELAY: return createDelay();
      case TextualSlcoPackage.PORT_REFERENCE: return createPortReference();
      case TextualSlcoPackage.SIGNAL_RECEPTION: return createSignalReception();
      case TextualSlcoPackage.SIGNAL_ARGUMENT: return createSignalArgument();
      case TextualSlcoPackage.SIGNAL_ARGUMENT_VARIABLE: return createSignalArgumentVariable();
      case TextualSlcoPackage.SIGNAL_ARGUMENT_EXPRESSION: return createSignalArgumentExpression();
      case TextualSlcoPackage.ASSIGNMENT: return createAssignment();
      case TextualSlcoPackage.SEND_SIGNAL: return createSendSignal();
      case TextualSlcoPackage.OBJECT: return createObject();
      case TextualSlcoPackage.PORT: return createPort();
      case TextualSlcoPackage.CHANNEL: return createChannel();
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL: return createUnidirectionalChannel();
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL: return createBidirectionalChannel();
      case TextualSlcoPackage.ARGUMENT_TYPE: return createArgumentType();
      case TextualSlcoPackage.VARIABLE: return createVariable();
      case TextualSlcoPackage.EXPRESSION: return createExpression();
      case TextualSlcoPackage.BOOLEAN_CONSTANT_EXPRESSION: return createBooleanConstantExpression();
      case TextualSlcoPackage.INTEGER_CONSTANT_EXPRESSION: return createIntegerConstantExpression();
      case TextualSlcoPackage.STRING_CONSTANT_EXPRESSION: return createStringConstantExpression();
      case TextualSlcoPackage.VARIABLE_REFERENCE: return createVariableReference();
      case TextualSlcoPackage.VARIABLE_EXPRESSION: return createVariableExpression();
      case TextualSlcoPackage.BINARY_OPERATOR_EXPRESSION: return createBinaryOperatorExpression();
      default:
        throw new IllegalArgumentException("The class '" + eClass.getName() + "' is not a valid classifier");
    }
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public Object createFromString(EDataType eDataType, String initialValue)
  {
    switch (eDataType.getClassifierID())
    {
      case TextualSlcoPackage.PRIMITIVE_TYPE:
        return createPrimitiveTypeFromString(eDataType, initialValue);
      case TextualSlcoPackage.CHANNEL_TYPE:
        return createChannelTypeFromString(eDataType, initialValue);
      case TextualSlcoPackage.OPERATOR:
        return createOperatorFromString(eDataType, initialValue);
      default:
        throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
    }
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public String convertToString(EDataType eDataType, Object instanceValue)
  {
    switch (eDataType.getClassifierID())
    {
      case TextualSlcoPackage.PRIMITIVE_TYPE:
        return convertPrimitiveTypeToString(eDataType, instanceValue);
      case TextualSlcoPackage.CHANNEL_TYPE:
        return convertChannelTypeToString(eDataType, instanceValue);
      case TextualSlcoPackage.OPERATOR:
        return convertOperatorToString(eDataType, instanceValue);
      default:
        throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
    }
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Model createModel()
  {
    ModelImpl model = new ModelImpl();
    return model;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public org.xtext.textualSlco.Class createClass()
  {
    ClassImpl class_ = new ClassImpl();
    return class_;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public StateMachine createStateMachine()
  {
    StateMachineImpl stateMachine = new StateMachineImpl();
    return stateMachine;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Vertex createVertex()
  {
    VertexImpl vertex = new VertexImpl();
    return vertex;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Initial createInitial()
  {
    InitialImpl initial = new InitialImpl();
    return initial;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public State createState()
  {
    StateImpl state = new StateImpl();
    return state;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Final createFinal()
  {
    FinalImpl final_ = new FinalImpl();
    return final_;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public VertexReference createVertexReference()
  {
    VertexReferenceImpl vertexReference = new VertexReferenceImpl();
    return vertexReference;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Transition createTransition()
  {
    TransitionImpl transition = new TransitionImpl();
    return transition;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Statement createStatement()
  {
    StatementImpl statement = new StatementImpl();
    return statement;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Delay createDelay()
  {
    DelayImpl delay = new DelayImpl();
    return delay;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public PortReference createPortReference()
  {
    PortReferenceImpl portReference = new PortReferenceImpl();
    return portReference;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public SignalReception createSignalReception()
  {
    SignalReceptionImpl signalReception = new SignalReceptionImpl();
    return signalReception;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public SignalArgument createSignalArgument()
  {
    SignalArgumentImpl signalArgument = new SignalArgumentImpl();
    return signalArgument;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public SignalArgumentVariable createSignalArgumentVariable()
  {
    SignalArgumentVariableImpl signalArgumentVariable = new SignalArgumentVariableImpl();
    return signalArgumentVariable;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public SignalArgumentExpression createSignalArgumentExpression()
  {
    SignalArgumentExpressionImpl signalArgumentExpression = new SignalArgumentExpressionImpl();
    return signalArgumentExpression;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Assignment createAssignment()
  {
    AssignmentImpl assignment = new AssignmentImpl();
    return assignment;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public SendSignal createSendSignal()
  {
    SendSignalImpl sendSignal = new SendSignalImpl();
    return sendSignal;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public org.xtext.textualSlco.Object createObject()
  {
    ObjectImpl object = new ObjectImpl();
    return object;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Port createPort()
  {
    PortImpl port = new PortImpl();
    return port;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Channel createChannel()
  {
    ChannelImpl channel = new ChannelImpl();
    return channel;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public UnidirectionalChannel createUnidirectionalChannel()
  {
    UnidirectionalChannelImpl unidirectionalChannel = new UnidirectionalChannelImpl();
    return unidirectionalChannel;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public BidirectionalChannel createBidirectionalChannel()
  {
    BidirectionalChannelImpl bidirectionalChannel = new BidirectionalChannelImpl();
    return bidirectionalChannel;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public ArgumentType createArgumentType()
  {
    ArgumentTypeImpl argumentType = new ArgumentTypeImpl();
    return argumentType;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Variable createVariable()
  {
    VariableImpl variable = new VariableImpl();
    return variable;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Expression createExpression()
  {
    ExpressionImpl expression = new ExpressionImpl();
    return expression;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public BooleanConstantExpression createBooleanConstantExpression()
  {
    BooleanConstantExpressionImpl booleanConstantExpression = new BooleanConstantExpressionImpl();
    return booleanConstantExpression;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public IntegerConstantExpression createIntegerConstantExpression()
  {
    IntegerConstantExpressionImpl integerConstantExpression = new IntegerConstantExpressionImpl();
    return integerConstantExpression;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public StringConstantExpression createStringConstantExpression()
  {
    StringConstantExpressionImpl stringConstantExpression = new StringConstantExpressionImpl();
    return stringConstantExpression;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public VariableReference createVariableReference()
  {
    VariableReferenceImpl variableReference = new VariableReferenceImpl();
    return variableReference;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public VariableExpression createVariableExpression()
  {
    VariableExpressionImpl variableExpression = new VariableExpressionImpl();
    return variableExpression;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public BinaryOperatorExpression createBinaryOperatorExpression()
  {
    BinaryOperatorExpressionImpl binaryOperatorExpression = new BinaryOperatorExpressionImpl();
    return binaryOperatorExpression;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public PrimitiveType createPrimitiveTypeFromString(EDataType eDataType, String initialValue)
  {
    PrimitiveType result = PrimitiveType.get(initialValue);
    if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
    return result;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public String convertPrimitiveTypeToString(EDataType eDataType, Object instanceValue)
  {
    return instanceValue == null ? null : instanceValue.toString();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public ChannelType createChannelTypeFromString(EDataType eDataType, String initialValue)
  {
    ChannelType result = ChannelType.get(initialValue);
    if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
    return result;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public String convertChannelTypeToString(EDataType eDataType, Object instanceValue)
  {
    return instanceValue == null ? null : instanceValue.toString();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Operator createOperatorFromString(EDataType eDataType, String initialValue)
  {
    Operator result = Operator.get(initialValue);
    if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
    return result;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public String convertOperatorToString(EDataType eDataType, Object instanceValue)
  {
    return instanceValue == null ? null : instanceValue.toString();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public TextualSlcoPackage getTextualSlcoPackage()
  {
    return (TextualSlcoPackage)getEPackage();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @deprecated
   * @generated
   */
  @Deprecated
  public static TextualSlcoPackage getPackage()
  {
    return TextualSlcoPackage.eINSTANCE;
  }

} //TextualSlcoFactoryImpl
