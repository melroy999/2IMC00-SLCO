/**
 */
package org.xtext.textualSlco;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @see org.xtext.textualSlco.TextualSlcoPackage
 * @generated
 */
public interface TextualSlcoFactory extends EFactory
{
  /**
   * The singleton instance of the factory.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  TextualSlcoFactory eINSTANCE = org.xtext.textualSlco.impl.TextualSlcoFactoryImpl.init();

  /**
   * Returns a new object of class '<em>Model</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Model</em>'.
   * @generated
   */
  Model createModel();

  /**
   * Returns a new object of class '<em>Class</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Class</em>'.
   * @generated
   */
  Class createClass();

  /**
   * Returns a new object of class '<em>State Machine</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>State Machine</em>'.
   * @generated
   */
  StateMachine createStateMachine();

  /**
   * Returns a new object of class '<em>Vertex</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Vertex</em>'.
   * @generated
   */
  Vertex createVertex();

  /**
   * Returns a new object of class '<em>Initial</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Initial</em>'.
   * @generated
   */
  Initial createInitial();

  /**
   * Returns a new object of class '<em>State</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>State</em>'.
   * @generated
   */
  State createState();

  /**
   * Returns a new object of class '<em>Final</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Final</em>'.
   * @generated
   */
  Final createFinal();

  /**
   * Returns a new object of class '<em>Vertex Reference</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Vertex Reference</em>'.
   * @generated
   */
  VertexReference createVertexReference();

  /**
   * Returns a new object of class '<em>Transition</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Transition</em>'.
   * @generated
   */
  Transition createTransition();

  /**
   * Returns a new object of class '<em>Statement</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Statement</em>'.
   * @generated
   */
  Statement createStatement();

  /**
   * Returns a new object of class '<em>Delay</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Delay</em>'.
   * @generated
   */
  Delay createDelay();

  /**
   * Returns a new object of class '<em>Port Reference</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Port Reference</em>'.
   * @generated
   */
  PortReference createPortReference();

  /**
   * Returns a new object of class '<em>Signal Reception</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Signal Reception</em>'.
   * @generated
   */
  SignalReception createSignalReception();

  /**
   * Returns a new object of class '<em>Signal Argument</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Signal Argument</em>'.
   * @generated
   */
  SignalArgument createSignalArgument();

  /**
   * Returns a new object of class '<em>Signal Argument Variable</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Signal Argument Variable</em>'.
   * @generated
   */
  SignalArgumentVariable createSignalArgumentVariable();

  /**
   * Returns a new object of class '<em>Signal Argument Expression</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Signal Argument Expression</em>'.
   * @generated
   */
  SignalArgumentExpression createSignalArgumentExpression();

  /**
   * Returns a new object of class '<em>Assignment</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Assignment</em>'.
   * @generated
   */
  Assignment createAssignment();

  /**
   * Returns a new object of class '<em>Send Signal</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Send Signal</em>'.
   * @generated
   */
  SendSignal createSendSignal();

  /**
   * Returns a new object of class '<em>Object</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Object</em>'.
   * @generated
   */
  Object createObject();

  /**
   * Returns a new object of class '<em>Port</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Port</em>'.
   * @generated
   */
  Port createPort();

  /**
   * Returns a new object of class '<em>Channel</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Channel</em>'.
   * @generated
   */
  Channel createChannel();

  /**
   * Returns a new object of class '<em>Unidirectional Channel</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Unidirectional Channel</em>'.
   * @generated
   */
  UnidirectionalChannel createUnidirectionalChannel();

  /**
   * Returns a new object of class '<em>Bidirectional Channel</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Bidirectional Channel</em>'.
   * @generated
   */
  BidirectionalChannel createBidirectionalChannel();

  /**
   * Returns a new object of class '<em>Argument Type</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Argument Type</em>'.
   * @generated
   */
  ArgumentType createArgumentType();

  /**
   * Returns a new object of class '<em>Variable</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Variable</em>'.
   * @generated
   */
  Variable createVariable();

  /**
   * Returns a new object of class '<em>Expression</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Expression</em>'.
   * @generated
   */
  Expression createExpression();

  /**
   * Returns a new object of class '<em>Boolean Constant Expression</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Boolean Constant Expression</em>'.
   * @generated
   */
  BooleanConstantExpression createBooleanConstantExpression();

  /**
   * Returns a new object of class '<em>Integer Constant Expression</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Integer Constant Expression</em>'.
   * @generated
   */
  IntegerConstantExpression createIntegerConstantExpression();

  /**
   * Returns a new object of class '<em>String Constant Expression</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>String Constant Expression</em>'.
   * @generated
   */
  StringConstantExpression createStringConstantExpression();

  /**
   * Returns a new object of class '<em>Variable Reference</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Variable Reference</em>'.
   * @generated
   */
  VariableReference createVariableReference();

  /**
   * Returns a new object of class '<em>Variable Expression</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Variable Expression</em>'.
   * @generated
   */
  VariableExpression createVariableExpression();

  /**
   * Returns a new object of class '<em>Binary Operator Expression</em>'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return a new object of class '<em>Binary Operator Expression</em>'.
   * @generated
   */
  BinaryOperatorExpression createBinaryOperatorExpression();

  /**
   * Returns the package supported by this factory.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the package supported by this factory.
   * @generated
   */
  TextualSlcoPackage getTextualSlcoPackage();

} //TextualSlcoFactory
