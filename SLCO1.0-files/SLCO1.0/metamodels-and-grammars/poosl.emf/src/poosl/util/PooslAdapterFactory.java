/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.util;

import org.eclipse.emf.common.notify.Adapter;
import org.eclipse.emf.common.notify.Notifier;

import org.eclipse.emf.common.notify.impl.AdapterFactoryImpl;

import org.eclipse.emf.ecore.EObject;

import poosl.Abort;
import poosl.Assignment;
import poosl.Atomic;
import poosl.Channel;
import poosl.ConditionalExpression;
import poosl.ConditionalStatement;
import poosl.Constant;
import poosl.ConstantExpression;
import poosl.CurrentTime;
import poosl.DataClass;
import poosl.DataMethod;
import poosl.DataMethodCall;
import poosl.DataObjectCreation;
import poosl.Expression;
import poosl.ExpressionList;
import poosl.GuardedExecution;
import poosl.IncomingMessage;
import poosl.Interrupt;
import poosl.LoopExpression;
import poosl.LoopStatement;
import poosl.Message;
import poosl.MessageReceive;
import poosl.MessageSend;
import poosl.Model;
import poosl.OutgoingMessage;
import poosl.ParallelComposition;
import poosl.Parameter;
import poosl.ParameterExpression;
import poosl.PooslPackage;
import poosl.Port;
import poosl.PrimitiveDataMethodCall;
import poosl.PrimitiveType;
import poosl.ProcessClass;
import poosl.ProcessInstance;
import poosl.ProcessMethod;
import poosl.ProcessMethodCall;
import poosl.ReferenceSelf;
import poosl.ReturnExpression;
import poosl.Selection;
import poosl.Skip;
import poosl.Statement;
import poosl.StatementList;
import poosl.TimeSynchronisation;
import poosl.Type;
import poosl.Variable;
import poosl.VariableExpression;

/**
 * <!-- begin-user-doc -->
 * The <b>Adapter Factory</b> for the model.
 * It provides an adapter <code>createXXX</code> method for each class of the model.
 * <!-- end-user-doc -->
 * @see poosl.PooslPackage
 * @generated
 */
public class PooslAdapterFactory extends AdapterFactoryImpl {
	/**
	 * The cached model package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static PooslPackage modelPackage;

	/**
	 * Creates an instance of the adapter factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PooslAdapterFactory() {
		if (modelPackage == null) {
			modelPackage = PooslPackage.eINSTANCE;
		}
	}

	/**
	 * Returns whether this factory is applicable for the type of the object.
	 * <!-- begin-user-doc -->
	 * This implementation returns <code>true</code> if the object is either the model's package or is an instance object of the model.
	 * <!-- end-user-doc -->
	 * @return whether this factory is applicable for the type of the object.
	 * @generated
	 */
	@Override
	public boolean isFactoryForType(Object object) {
		if (object == modelPackage) {
			return true;
		}
		if (object instanceof EObject) {
			return ((EObject)object).eClass().getEPackage() == modelPackage;
		}
		return false;
	}

	/**
	 * The switch that delegates to the <code>createXXX</code> methods.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected PooslSwitch<Adapter> modelSwitch =
		new PooslSwitch<Adapter>() {
			@Override
			public Adapter caseAbort(Abort object) {
				return createAbortAdapter();
			}
			@Override
			public Adapter caseAssignment(Assignment object) {
				return createAssignmentAdapter();
			}
			@Override
			public Adapter caseAtomic(Atomic object) {
				return createAtomicAdapter();
			}
			@Override
			public Adapter caseChannel(Channel object) {
				return createChannelAdapter();
			}
			@Override
			public Adapter caseClass(poosl.Class object) {
				return createClassAdapter();
			}
			@Override
			public Adapter caseConditionalExpression(ConditionalExpression object) {
				return createConditionalExpressionAdapter();
			}
			@Override
			public Adapter caseConditionalStatement(ConditionalStatement object) {
				return createConditionalStatementAdapter();
			}
			@Override
			public Adapter caseConstant(Constant object) {
				return createConstantAdapter();
			}
			@Override
			public Adapter caseConstantExpression(ConstantExpression object) {
				return createConstantExpressionAdapter();
			}
			@Override
			public Adapter caseCurrentTime(CurrentTime object) {
				return createCurrentTimeAdapter();
			}
			@Override
			public Adapter caseDataClass(DataClass object) {
				return createDataClassAdapter();
			}
			@Override
			public Adapter caseDataMethod(DataMethod object) {
				return createDataMethodAdapter();
			}
			@Override
			public Adapter caseDataMethodCall(DataMethodCall object) {
				return createDataMethodCallAdapter();
			}
			@Override
			public Adapter casePrimitiveDataMethodCall(PrimitiveDataMethodCall object) {
				return createPrimitiveDataMethodCallAdapter();
			}
			@Override
			public Adapter caseDataObjectCreation(DataObjectCreation object) {
				return createDataObjectCreationAdapter();
			}
			@Override
			public Adapter caseExpression(Expression object) {
				return createExpressionAdapter();
			}
			@Override
			public Adapter caseExpressionList(ExpressionList object) {
				return createExpressionListAdapter();
			}
			@Override
			public Adapter caseGuardedExecution(GuardedExecution object) {
				return createGuardedExecutionAdapter();
			}
			@Override
			public Adapter caseIncomingMessage(IncomingMessage object) {
				return createIncomingMessageAdapter();
			}
			@Override
			public Adapter caseInterrupt(Interrupt object) {
				return createInterruptAdapter();
			}
			@Override
			public Adapter caseLoopExpression(LoopExpression object) {
				return createLoopExpressionAdapter();
			}
			@Override
			public Adapter caseLoopStatement(LoopStatement object) {
				return createLoopStatementAdapter();
			}
			@Override
			public Adapter caseMessage(Message object) {
				return createMessageAdapter();
			}
			@Override
			public Adapter caseMessageReceive(MessageReceive object) {
				return createMessageReceiveAdapter();
			}
			@Override
			public Adapter caseMessageSend(MessageSend object) {
				return createMessageSendAdapter();
			}
			@Override
			public Adapter caseModel(Model object) {
				return createModelAdapter();
			}
			@Override
			public Adapter caseOutgoingMessage(OutgoingMessage object) {
				return createOutgoingMessageAdapter();
			}
			@Override
			public Adapter caseParallelComposition(ParallelComposition object) {
				return createParallelCompositionAdapter();
			}
			@Override
			public Adapter caseParameter(Parameter object) {
				return createParameterAdapter();
			}
			@Override
			public Adapter caseParameterExpression(ParameterExpression object) {
				return createParameterExpressionAdapter();
			}
			@Override
			public Adapter casePrimitiveType(PrimitiveType object) {
				return createPrimitiveTypeAdapter();
			}
			@Override
			public Adapter caseProcessClass(ProcessClass object) {
				return createProcessClassAdapter();
			}
			@Override
			public Adapter caseProcessInstance(ProcessInstance object) {
				return createProcessInstanceAdapter();
			}
			@Override
			public Adapter caseProcessMethod(ProcessMethod object) {
				return createProcessMethodAdapter();
			}
			@Override
			public Adapter caseProcessMethodCall(ProcessMethodCall object) {
				return createProcessMethodCallAdapter();
			}
			@Override
			public Adapter casePort(Port object) {
				return createPortAdapter();
			}
			@Override
			public Adapter caseReferenceSelf(ReferenceSelf object) {
				return createReferenceSelfAdapter();
			}
			@Override
			public Adapter caseReturnExpression(ReturnExpression object) {
				return createReturnExpressionAdapter();
			}
			@Override
			public Adapter caseSelection(Selection object) {
				return createSelectionAdapter();
			}
			@Override
			public Adapter caseSkip(Skip object) {
				return createSkipAdapter();
			}
			@Override
			public Adapter caseStatement(Statement object) {
				return createStatementAdapter();
			}
			@Override
			public Adapter caseStatementList(StatementList object) {
				return createStatementListAdapter();
			}
			@Override
			public Adapter caseTimeSynchronisation(TimeSynchronisation object) {
				return createTimeSynchronisationAdapter();
			}
			@Override
			public Adapter caseType(Type object) {
				return createTypeAdapter();
			}
			@Override
			public Adapter caseVariable(Variable object) {
				return createVariableAdapter();
			}
			@Override
			public Adapter caseVariableExpression(VariableExpression object) {
				return createVariableExpressionAdapter();
			}
			@Override
			public Adapter defaultCase(EObject object) {
				return createEObjectAdapter();
			}
		};

	/**
	 * Creates an adapter for the <code>target</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param target the object to adapt.
	 * @return the adapter for the <code>target</code>.
	 * @generated
	 */
	@Override
	public Adapter createAdapter(Notifier target) {
		return modelSwitch.doSwitch((EObject)target);
	}


	/**
	 * Creates a new adapter for an object of class '{@link poosl.Abort <em>Abort</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Abort
	 * @generated
	 */
	public Adapter createAbortAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Assignment <em>Assignment</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Assignment
	 * @generated
	 */
	public Adapter createAssignmentAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Atomic <em>Atomic</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Atomic
	 * @generated
	 */
	public Adapter createAtomicAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Channel <em>Channel</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Channel
	 * @generated
	 */
	public Adapter createChannelAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Class <em>Class</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Class
	 * @generated
	 */
	public Adapter createClassAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ConditionalExpression <em>Conditional Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ConditionalExpression
	 * @generated
	 */
	public Adapter createConditionalExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ConditionalStatement <em>Conditional Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ConditionalStatement
	 * @generated
	 */
	public Adapter createConditionalStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Constant <em>Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Constant
	 * @generated
	 */
	public Adapter createConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ConstantExpression <em>Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ConstantExpression
	 * @generated
	 */
	public Adapter createConstantExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.CurrentTime <em>Current Time</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.CurrentTime
	 * @generated
	 */
	public Adapter createCurrentTimeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.DataClass <em>Data Class</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.DataClass
	 * @generated
	 */
	public Adapter createDataClassAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.DataMethod <em>Data Method</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.DataMethod
	 * @generated
	 */
	public Adapter createDataMethodAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.DataMethodCall <em>Data Method Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.DataMethodCall
	 * @generated
	 */
	public Adapter createDataMethodCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.PrimitiveDataMethodCall <em>Primitive Data Method Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.PrimitiveDataMethodCall
	 * @generated
	 */
	public Adapter createPrimitiveDataMethodCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.DataObjectCreation <em>Data Object Creation</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.DataObjectCreation
	 * @generated
	 */
	public Adapter createDataObjectCreationAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Expression <em>Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Expression
	 * @generated
	 */
	public Adapter createExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ExpressionList <em>Expression List</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ExpressionList
	 * @generated
	 */
	public Adapter createExpressionListAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.GuardedExecution <em>Guarded Execution</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.GuardedExecution
	 * @generated
	 */
	public Adapter createGuardedExecutionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.IncomingMessage <em>Incoming Message</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.IncomingMessage
	 * @generated
	 */
	public Adapter createIncomingMessageAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Interrupt <em>Interrupt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Interrupt
	 * @generated
	 */
	public Adapter createInterruptAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.LoopExpression <em>Loop Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.LoopExpression
	 * @generated
	 */
	public Adapter createLoopExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.LoopStatement <em>Loop Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.LoopStatement
	 * @generated
	 */
	public Adapter createLoopStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Message <em>Message</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Message
	 * @generated
	 */
	public Adapter createMessageAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.MessageReceive <em>Message Receive</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.MessageReceive
	 * @generated
	 */
	public Adapter createMessageReceiveAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.MessageSend <em>Message Send</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.MessageSend
	 * @generated
	 */
	public Adapter createMessageSendAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Model <em>Model</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Model
	 * @generated
	 */
	public Adapter createModelAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.OutgoingMessage <em>Outgoing Message</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.OutgoingMessage
	 * @generated
	 */
	public Adapter createOutgoingMessageAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ParallelComposition <em>Parallel Composition</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ParallelComposition
	 * @generated
	 */
	public Adapter createParallelCompositionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Parameter <em>Parameter</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Parameter
	 * @generated
	 */
	public Adapter createParameterAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ParameterExpression <em>Parameter Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ParameterExpression
	 * @generated
	 */
	public Adapter createParameterExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.PrimitiveType <em>Primitive Type</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.PrimitiveType
	 * @generated
	 */
	public Adapter createPrimitiveTypeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ProcessClass <em>Process Class</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ProcessClass
	 * @generated
	 */
	public Adapter createProcessClassAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ProcessInstance <em>Process Instance</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ProcessInstance
	 * @generated
	 */
	public Adapter createProcessInstanceAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ProcessMethod <em>Process Method</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ProcessMethod
	 * @generated
	 */
	public Adapter createProcessMethodAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ProcessMethodCall <em>Process Method Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ProcessMethodCall
	 * @generated
	 */
	public Adapter createProcessMethodCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Port <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Port
	 * @generated
	 */
	public Adapter createPortAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ReferenceSelf <em>Reference Self</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ReferenceSelf
	 * @generated
	 */
	public Adapter createReferenceSelfAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.ReturnExpression <em>Return Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.ReturnExpression
	 * @generated
	 */
	public Adapter createReturnExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Selection <em>Selection</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Selection
	 * @generated
	 */
	public Adapter createSelectionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Skip <em>Skip</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Skip
	 * @generated
	 */
	public Adapter createSkipAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Statement <em>Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Statement
	 * @generated
	 */
	public Adapter createStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.StatementList <em>Statement List</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.StatementList
	 * @generated
	 */
	public Adapter createStatementListAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.TimeSynchronisation <em>Time Synchronisation</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.TimeSynchronisation
	 * @generated
	 */
	public Adapter createTimeSynchronisationAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Type <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Type
	 * @generated
	 */
	public Adapter createTypeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.Variable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.Variable
	 * @generated
	 */
	public Adapter createVariableAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link poosl.VariableExpression <em>Variable Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see poosl.VariableExpression
	 * @generated
	 */
	public Adapter createVariableExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for the default case.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @generated
	 */
	public Adapter createEObjectAdapter() {
		return null;
	}

} //PooslAdapterFactory
