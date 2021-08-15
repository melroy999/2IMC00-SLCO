/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @see poosl.PooslPackage
 * @generated
 */
public interface PooslFactory extends EFactory {
	/**
	 * The singleton instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	PooslFactory eINSTANCE = poosl.impl.PooslFactoryImpl.init();

	/**
	 * Returns a new object of class '<em>Abort</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Abort</em>'.
	 * @generated
	 */
	Abort createAbort();

	/**
	 * Returns a new object of class '<em>Assignment</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Assignment</em>'.
	 * @generated
	 */
	Assignment createAssignment();

	/**
	 * Returns a new object of class '<em>Atomic</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Atomic</em>'.
	 * @generated
	 */
	Atomic createAtomic();

	/**
	 * Returns a new object of class '<em>Channel</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Channel</em>'.
	 * @generated
	 */
	Channel createChannel();

	/**
	 * Returns a new object of class '<em>Conditional Expression</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Conditional Expression</em>'.
	 * @generated
	 */
	ConditionalExpression createConditionalExpression();

	/**
	 * Returns a new object of class '<em>Conditional Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Conditional Statement</em>'.
	 * @generated
	 */
	ConditionalStatement createConditionalStatement();

	/**
	 * Returns a new object of class '<em>Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Constant</em>'.
	 * @generated
	 */
	Constant createConstant();

	/**
	 * Returns a new object of class '<em>Constant Expression</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Constant Expression</em>'.
	 * @generated
	 */
	ConstantExpression createConstantExpression();

	/**
	 * Returns a new object of class '<em>Current Time</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Current Time</em>'.
	 * @generated
	 */
	CurrentTime createCurrentTime();

	/**
	 * Returns a new object of class '<em>Data Class</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Data Class</em>'.
	 * @generated
	 */
	DataClass createDataClass();

	/**
	 * Returns a new object of class '<em>Data Method</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Data Method</em>'.
	 * @generated
	 */
	DataMethod createDataMethod();

	/**
	 * Returns a new object of class '<em>Data Method Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Data Method Call</em>'.
	 * @generated
	 */
	DataMethodCall createDataMethodCall();

	/**
	 * Returns a new object of class '<em>Primitive Data Method Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Primitive Data Method Call</em>'.
	 * @generated
	 */
	PrimitiveDataMethodCall createPrimitiveDataMethodCall();

	/**
	 * Returns a new object of class '<em>Data Object Creation</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Data Object Creation</em>'.
	 * @generated
	 */
	DataObjectCreation createDataObjectCreation();

	/**
	 * Returns a new object of class '<em>Expression List</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Expression List</em>'.
	 * @generated
	 */
	ExpressionList createExpressionList();

	/**
	 * Returns a new object of class '<em>Guarded Execution</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Guarded Execution</em>'.
	 * @generated
	 */
	GuardedExecution createGuardedExecution();

	/**
	 * Returns a new object of class '<em>Incoming Message</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Incoming Message</em>'.
	 * @generated
	 */
	IncomingMessage createIncomingMessage();

	/**
	 * Returns a new object of class '<em>Interrupt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Interrupt</em>'.
	 * @generated
	 */
	Interrupt createInterrupt();

	/**
	 * Returns a new object of class '<em>Loop Expression</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Loop Expression</em>'.
	 * @generated
	 */
	LoopExpression createLoopExpression();

	/**
	 * Returns a new object of class '<em>Loop Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Loop Statement</em>'.
	 * @generated
	 */
	LoopStatement createLoopStatement();

	/**
	 * Returns a new object of class '<em>Message</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Message</em>'.
	 * @generated
	 */
	Message createMessage();

	/**
	 * Returns a new object of class '<em>Message Receive</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Message Receive</em>'.
	 * @generated
	 */
	MessageReceive createMessageReceive();

	/**
	 * Returns a new object of class '<em>Message Send</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Message Send</em>'.
	 * @generated
	 */
	MessageSend createMessageSend();

	/**
	 * Returns a new object of class '<em>Model</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Model</em>'.
	 * @generated
	 */
	Model createModel();

	/**
	 * Returns a new object of class '<em>Outgoing Message</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Outgoing Message</em>'.
	 * @generated
	 */
	OutgoingMessage createOutgoingMessage();

	/**
	 * Returns a new object of class '<em>Parallel Composition</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Parallel Composition</em>'.
	 * @generated
	 */
	ParallelComposition createParallelComposition();

	/**
	 * Returns a new object of class '<em>Parameter</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Parameter</em>'.
	 * @generated
	 */
	Parameter createParameter();

	/**
	 * Returns a new object of class '<em>Parameter Expression</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Parameter Expression</em>'.
	 * @generated
	 */
	ParameterExpression createParameterExpression();

	/**
	 * Returns a new object of class '<em>Primitive Type</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Primitive Type</em>'.
	 * @generated
	 */
	PrimitiveType createPrimitiveType();

	/**
	 * Returns a new object of class '<em>Process Class</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Process Class</em>'.
	 * @generated
	 */
	ProcessClass createProcessClass();

	/**
	 * Returns a new object of class '<em>Process Instance</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Process Instance</em>'.
	 * @generated
	 */
	ProcessInstance createProcessInstance();

	/**
	 * Returns a new object of class '<em>Process Method</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Process Method</em>'.
	 * @generated
	 */
	ProcessMethod createProcessMethod();

	/**
	 * Returns a new object of class '<em>Process Method Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Process Method Call</em>'.
	 * @generated
	 */
	ProcessMethodCall createProcessMethodCall();

	/**
	 * Returns a new object of class '<em>Port</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Port</em>'.
	 * @generated
	 */
	Port createPort();

	/**
	 * Returns a new object of class '<em>Reference Self</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Reference Self</em>'.
	 * @generated
	 */
	ReferenceSelf createReferenceSelf();

	/**
	 * Returns a new object of class '<em>Return Expression</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Return Expression</em>'.
	 * @generated
	 */
	ReturnExpression createReturnExpression();

	/**
	 * Returns a new object of class '<em>Selection</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Selection</em>'.
	 * @generated
	 */
	Selection createSelection();

	/**
	 * Returns a new object of class '<em>Skip</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Skip</em>'.
	 * @generated
	 */
	Skip createSkip();

	/**
	 * Returns a new object of class '<em>Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Statement</em>'.
	 * @generated
	 */
	Statement createStatement();

	/**
	 * Returns a new object of class '<em>Statement List</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Statement List</em>'.
	 * @generated
	 */
	StatementList createStatementList();

	/**
	 * Returns a new object of class '<em>Time Synchronisation</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Time Synchronisation</em>'.
	 * @generated
	 */
	TimeSynchronisation createTimeSynchronisation();

	/**
	 * Returns a new object of class '<em>Variable</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Variable</em>'.
	 * @generated
	 */
	Variable createVariable();

	/**
	 * Returns a new object of class '<em>Variable Expression</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Variable Expression</em>'.
	 * @generated
	 */
	VariableExpression createVariableExpression();

	/**
	 * Returns the package supported by this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package supported by this factory.
	 * @generated
	 */
	PooslPackage getPooslPackage();

} //PooslFactory
