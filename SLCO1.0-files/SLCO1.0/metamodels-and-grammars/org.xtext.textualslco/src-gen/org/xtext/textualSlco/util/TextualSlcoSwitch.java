/**
 */
package org.xtext.textualSlco.util;

import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.util.Switch;

import org.xtext.textualSlco.ArgumentType;
import org.xtext.textualSlco.Assignment;
import org.xtext.textualSlco.BidirectionalChannel;
import org.xtext.textualSlco.BinaryOperatorExpression;
import org.xtext.textualSlco.BooleanConstantExpression;
import org.xtext.textualSlco.Channel;
import org.xtext.textualSlco.Delay;
import org.xtext.textualSlco.Expression;
import org.xtext.textualSlco.Final;
import org.xtext.textualSlco.Initial;
import org.xtext.textualSlco.IntegerConstantExpression;
import org.xtext.textualSlco.Model;
import org.xtext.textualSlco.Port;
import org.xtext.textualSlco.PortReference;
import org.xtext.textualSlco.SendSignal;
import org.xtext.textualSlco.SignalArgument;
import org.xtext.textualSlco.SignalArgumentExpression;
import org.xtext.textualSlco.SignalArgumentVariable;
import org.xtext.textualSlco.SignalReception;
import org.xtext.textualSlco.State;
import org.xtext.textualSlco.StateMachine;
import org.xtext.textualSlco.Statement;
import org.xtext.textualSlco.StringConstantExpression;
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
 * The <b>Switch</b> for the model's inheritance hierarchy.
 * It supports the call {@link #doSwitch(EObject) doSwitch(object)}
 * to invoke the <code>caseXXX</code> method for each class of the model,
 * starting with the actual class of the object
 * and proceeding up the inheritance hierarchy
 * until a non-null result is returned,
 * which is the result of the switch.
 * <!-- end-user-doc -->
 * @see org.xtext.textualSlco.TextualSlcoPackage
 * @generated
 */
public class TextualSlcoSwitch<T> extends Switch<T>
{
  /**
   * The cached model package
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected static TextualSlcoPackage modelPackage;

  /**
   * Creates an instance of the switch.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public TextualSlcoSwitch()
  {
    if (modelPackage == null)
    {
      modelPackage = TextualSlcoPackage.eINSTANCE;
    }
  }

  /**
   * Checks whether this is a switch for the given package.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @parameter ePackage the package in question.
   * @return whether this is a switch for the given package.
   * @generated
   */
  @Override
  protected boolean isSwitchFor(EPackage ePackage)
  {
    return ePackage == modelPackage;
  }

  /**
   * Calls <code>caseXXX</code> for each class of the model until one returns a non null result; it yields that result.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the first non-null result returned by a <code>caseXXX</code> call.
   * @generated
   */
  @Override
  protected T doSwitch(int classifierID, EObject theEObject)
  {
    switch (classifierID)
    {
      case TextualSlcoPackage.MODEL:
      {
        Model model = (Model)theEObject;
        T result = caseModel(model);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.CLASS:
      {
        org.xtext.textualSlco.Class class_ = (org.xtext.textualSlco.Class)theEObject;
        T result = caseClass(class_);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.STATE_MACHINE:
      {
        StateMachine stateMachine = (StateMachine)theEObject;
        T result = caseStateMachine(stateMachine);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.VERTEX:
      {
        Vertex vertex = (Vertex)theEObject;
        T result = caseVertex(vertex);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.INITIAL:
      {
        Initial initial = (Initial)theEObject;
        T result = caseInitial(initial);
        if (result == null) result = caseVertex(initial);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.STATE:
      {
        State state = (State)theEObject;
        T result = caseState(state);
        if (result == null) result = caseVertex(state);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.FINAL:
      {
        Final final_ = (Final)theEObject;
        T result = caseFinal(final_);
        if (result == null) result = caseVertex(final_);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.VERTEX_REFERENCE:
      {
        VertexReference vertexReference = (VertexReference)theEObject;
        T result = caseVertexReference(vertexReference);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.TRANSITION:
      {
        Transition transition = (Transition)theEObject;
        T result = caseTransition(transition);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.STATEMENT:
      {
        Statement statement = (Statement)theEObject;
        T result = caseStatement(statement);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.DELAY:
      {
        Delay delay = (Delay)theEObject;
        T result = caseDelay(delay);
        if (result == null) result = caseStatement(delay);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.PORT_REFERENCE:
      {
        PortReference portReference = (PortReference)theEObject;
        T result = casePortReference(portReference);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.SIGNAL_RECEPTION:
      {
        SignalReception signalReception = (SignalReception)theEObject;
        T result = caseSignalReception(signalReception);
        if (result == null) result = caseStatement(signalReception);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.SIGNAL_ARGUMENT:
      {
        SignalArgument signalArgument = (SignalArgument)theEObject;
        T result = caseSignalArgument(signalArgument);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.SIGNAL_ARGUMENT_VARIABLE:
      {
        SignalArgumentVariable signalArgumentVariable = (SignalArgumentVariable)theEObject;
        T result = caseSignalArgumentVariable(signalArgumentVariable);
        if (result == null) result = caseSignalArgument(signalArgumentVariable);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.SIGNAL_ARGUMENT_EXPRESSION:
      {
        SignalArgumentExpression signalArgumentExpression = (SignalArgumentExpression)theEObject;
        T result = caseSignalArgumentExpression(signalArgumentExpression);
        if (result == null) result = caseSignalArgument(signalArgumentExpression);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.ASSIGNMENT:
      {
        Assignment assignment = (Assignment)theEObject;
        T result = caseAssignment(assignment);
        if (result == null) result = caseStatement(assignment);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.SEND_SIGNAL:
      {
        SendSignal sendSignal = (SendSignal)theEObject;
        T result = caseSendSignal(sendSignal);
        if (result == null) result = caseStatement(sendSignal);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.OBJECT:
      {
        org.xtext.textualSlco.Object object = (org.xtext.textualSlco.Object)theEObject;
        T result = caseObject(object);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.PORT:
      {
        Port port = (Port)theEObject;
        T result = casePort(port);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.CHANNEL:
      {
        Channel channel = (Channel)theEObject;
        T result = caseChannel(channel);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL:
      {
        UnidirectionalChannel unidirectionalChannel = (UnidirectionalChannel)theEObject;
        T result = caseUnidirectionalChannel(unidirectionalChannel);
        if (result == null) result = caseChannel(unidirectionalChannel);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL:
      {
        BidirectionalChannel bidirectionalChannel = (BidirectionalChannel)theEObject;
        T result = caseBidirectionalChannel(bidirectionalChannel);
        if (result == null) result = caseChannel(bidirectionalChannel);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.ARGUMENT_TYPE:
      {
        ArgumentType argumentType = (ArgumentType)theEObject;
        T result = caseArgumentType(argumentType);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.VARIABLE:
      {
        Variable variable = (Variable)theEObject;
        T result = caseVariable(variable);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.EXPRESSION:
      {
        Expression expression = (Expression)theEObject;
        T result = caseExpression(expression);
        if (result == null) result = caseStatement(expression);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.BOOLEAN_CONSTANT_EXPRESSION:
      {
        BooleanConstantExpression booleanConstantExpression = (BooleanConstantExpression)theEObject;
        T result = caseBooleanConstantExpression(booleanConstantExpression);
        if (result == null) result = caseExpression(booleanConstantExpression);
        if (result == null) result = caseStatement(booleanConstantExpression);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.INTEGER_CONSTANT_EXPRESSION:
      {
        IntegerConstantExpression integerConstantExpression = (IntegerConstantExpression)theEObject;
        T result = caseIntegerConstantExpression(integerConstantExpression);
        if (result == null) result = caseExpression(integerConstantExpression);
        if (result == null) result = caseStatement(integerConstantExpression);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.STRING_CONSTANT_EXPRESSION:
      {
        StringConstantExpression stringConstantExpression = (StringConstantExpression)theEObject;
        T result = caseStringConstantExpression(stringConstantExpression);
        if (result == null) result = caseExpression(stringConstantExpression);
        if (result == null) result = caseStatement(stringConstantExpression);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.VARIABLE_REFERENCE:
      {
        VariableReference variableReference = (VariableReference)theEObject;
        T result = caseVariableReference(variableReference);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.VARIABLE_EXPRESSION:
      {
        VariableExpression variableExpression = (VariableExpression)theEObject;
        T result = caseVariableExpression(variableExpression);
        if (result == null) result = caseExpression(variableExpression);
        if (result == null) result = caseStatement(variableExpression);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      case TextualSlcoPackage.BINARY_OPERATOR_EXPRESSION:
      {
        BinaryOperatorExpression binaryOperatorExpression = (BinaryOperatorExpression)theEObject;
        T result = caseBinaryOperatorExpression(binaryOperatorExpression);
        if (result == null) result = caseExpression(binaryOperatorExpression);
        if (result == null) result = caseStatement(binaryOperatorExpression);
        if (result == null) result = defaultCase(theEObject);
        return result;
      }
      default: return defaultCase(theEObject);
    }
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Model</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Model</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseModel(Model object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Class</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Class</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseClass(org.xtext.textualSlco.Class object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>State Machine</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>State Machine</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseStateMachine(StateMachine object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Vertex</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Vertex</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseVertex(Vertex object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Initial</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Initial</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseInitial(Initial object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>State</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>State</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseState(State object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Final</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Final</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseFinal(Final object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Vertex Reference</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Vertex Reference</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseVertexReference(VertexReference object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Transition</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Transition</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseTransition(Transition object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Statement</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Statement</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseStatement(Statement object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Delay</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Delay</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseDelay(Delay object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Port Reference</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Port Reference</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T casePortReference(PortReference object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Signal Reception</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Signal Reception</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseSignalReception(SignalReception object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Signal Argument</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Signal Argument</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseSignalArgument(SignalArgument object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Signal Argument Variable</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Signal Argument Variable</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseSignalArgumentVariable(SignalArgumentVariable object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Signal Argument Expression</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Signal Argument Expression</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseSignalArgumentExpression(SignalArgumentExpression object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Assignment</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Assignment</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseAssignment(Assignment object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Send Signal</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Send Signal</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseSendSignal(SendSignal object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Object</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Object</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseObject(org.xtext.textualSlco.Object object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Port</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Port</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T casePort(Port object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Channel</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Channel</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseChannel(Channel object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Unidirectional Channel</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Unidirectional Channel</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseUnidirectionalChannel(UnidirectionalChannel object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Bidirectional Channel</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Bidirectional Channel</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseBidirectionalChannel(BidirectionalChannel object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Argument Type</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Argument Type</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseArgumentType(ArgumentType object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Variable</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Variable</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseVariable(Variable object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Expression</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Expression</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseExpression(Expression object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Boolean Constant Expression</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Boolean Constant Expression</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseBooleanConstantExpression(BooleanConstantExpression object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Integer Constant Expression</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Integer Constant Expression</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseIntegerConstantExpression(IntegerConstantExpression object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>String Constant Expression</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>String Constant Expression</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseStringConstantExpression(StringConstantExpression object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Variable Reference</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Variable Reference</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseVariableReference(VariableReference object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Variable Expression</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Variable Expression</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseVariableExpression(VariableExpression object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>Binary Operator Expression</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>Binary Operator Expression</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
   * @generated
   */
  public T caseBinaryOperatorExpression(BinaryOperatorExpression object)
  {
    return null;
  }

  /**
   * Returns the result of interpreting the object as an instance of '<em>EObject</em>'.
   * <!-- begin-user-doc -->
   * This implementation returns null;
   * returning a non-null result will terminate the switch, but this is the last case anyway.
   * <!-- end-user-doc -->
   * @param object the target of the switch.
   * @return the result of interpreting the object as an instance of '<em>EObject</em>'.
   * @see #doSwitch(org.eclipse.emf.ecore.EObject)
   * @generated
   */
  @Override
  public T defaultCase(EObject object)
  {
    return null;
  }

} //TextualSlcoSwitch
