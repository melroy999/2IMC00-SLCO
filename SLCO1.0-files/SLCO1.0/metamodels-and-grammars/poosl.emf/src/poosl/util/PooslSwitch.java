/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.util;

import java.util.List;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;

import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.util.Switch;
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
 * The <b>Switch</b> for the model's inheritance hierarchy.
 * It supports the call {@link #doSwitch(EObject) doSwitch(object)}
 * to invoke the <code>caseXXX</code> method for each class of the model,
 * starting with the actual class of the object
 * and proceeding up the inheritance hierarchy
 * until a non-null result is returned,
 * which is the result of the switch.
 * <!-- end-user-doc -->
 * @see poosl.PooslPackage
 * @generated
 */
public class PooslSwitch<T> extends Switch<T> {
	/**
	 * The cached model package
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static PooslPackage modelPackage;

	/**
	 * Creates an instance of the switch.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PooslSwitch() {
		if (modelPackage == null) {
			modelPackage = PooslPackage.eINSTANCE;
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
	protected boolean isSwitchFor(EPackage ePackage) {
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
	protected T doSwitch(int classifierID, EObject theEObject) {
		switch (classifierID) {
			case PooslPackage.ABORT: {
				Abort abort = (Abort)theEObject;
				T result = caseAbort(abort);
				if (result == null) result = caseStatement(abort);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.ASSIGNMENT: {
				Assignment assignment = (Assignment)theEObject;
				T result = caseAssignment(assignment);
				if (result == null) result = caseExpression(assignment);
				if (result == null) result = caseStatement(assignment);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.ATOMIC: {
				Atomic atomic = (Atomic)theEObject;
				T result = caseAtomic(atomic);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.CHANNEL: {
				Channel channel = (Channel)theEObject;
				T result = caseChannel(channel);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.CLASS: {
				poosl.Class class_ = (poosl.Class)theEObject;
				T result = caseClass(class_);
				if (result == null) result = caseType(class_);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.CONDITIONAL_EXPRESSION: {
				ConditionalExpression conditionalExpression = (ConditionalExpression)theEObject;
				T result = caseConditionalExpression(conditionalExpression);
				if (result == null) result = caseExpression(conditionalExpression);
				if (result == null) result = caseStatement(conditionalExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.CONDITIONAL_STATEMENT: {
				ConditionalStatement conditionalStatement = (ConditionalStatement)theEObject;
				T result = caseConditionalStatement(conditionalStatement);
				if (result == null) result = caseStatement(conditionalStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.CONSTANT: {
				Constant constant = (Constant)theEObject;
				T result = caseConstant(constant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.CONSTANT_EXPRESSION: {
				ConstantExpression constantExpression = (ConstantExpression)theEObject;
				T result = caseConstantExpression(constantExpression);
				if (result == null) result = caseExpression(constantExpression);
				if (result == null) result = caseStatement(constantExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.CURRENT_TIME: {
				CurrentTime currentTime = (CurrentTime)theEObject;
				T result = caseCurrentTime(currentTime);
				if (result == null) result = caseExpression(currentTime);
				if (result == null) result = caseStatement(currentTime);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.DATA_CLASS: {
				DataClass dataClass = (DataClass)theEObject;
				T result = caseDataClass(dataClass);
				if (result == null) result = caseClass(dataClass);
				if (result == null) result = caseType(dataClass);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.DATA_METHOD: {
				DataMethod dataMethod = (DataMethod)theEObject;
				T result = caseDataMethod(dataMethod);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.DATA_METHOD_CALL: {
				DataMethodCall dataMethodCall = (DataMethodCall)theEObject;
				T result = caseDataMethodCall(dataMethodCall);
				if (result == null) result = caseExpression(dataMethodCall);
				if (result == null) result = caseStatement(dataMethodCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL: {
				PrimitiveDataMethodCall primitiveDataMethodCall = (PrimitiveDataMethodCall)theEObject;
				T result = casePrimitiveDataMethodCall(primitiveDataMethodCall);
				if (result == null) result = caseExpression(primitiveDataMethodCall);
				if (result == null) result = caseStatement(primitiveDataMethodCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.DATA_OBJECT_CREATION: {
				DataObjectCreation dataObjectCreation = (DataObjectCreation)theEObject;
				T result = caseDataObjectCreation(dataObjectCreation);
				if (result == null) result = caseExpression(dataObjectCreation);
				if (result == null) result = caseStatement(dataObjectCreation);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.EXPRESSION: {
				Expression expression = (Expression)theEObject;
				T result = caseExpression(expression);
				if (result == null) result = caseStatement(expression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.EXPRESSION_LIST: {
				ExpressionList expressionList = (ExpressionList)theEObject;
				T result = caseExpressionList(expressionList);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.GUARDED_EXECUTION: {
				GuardedExecution guardedExecution = (GuardedExecution)theEObject;
				T result = caseGuardedExecution(guardedExecution);
				if (result == null) result = caseStatement(guardedExecution);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.INCOMING_MESSAGE: {
				IncomingMessage incomingMessage = (IncomingMessage)theEObject;
				T result = caseIncomingMessage(incomingMessage);
				if (result == null) result = caseMessage(incomingMessage);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.INTERRUPT: {
				Interrupt interrupt = (Interrupt)theEObject;
				T result = caseInterrupt(interrupt);
				if (result == null) result = caseStatement(interrupt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.LOOP_EXPRESSION: {
				LoopExpression loopExpression = (LoopExpression)theEObject;
				T result = caseLoopExpression(loopExpression);
				if (result == null) result = caseExpression(loopExpression);
				if (result == null) result = caseStatement(loopExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.LOOP_STATEMENT: {
				LoopStatement loopStatement = (LoopStatement)theEObject;
				T result = caseLoopStatement(loopStatement);
				if (result == null) result = caseStatement(loopStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.MESSAGE: {
				Message message = (Message)theEObject;
				T result = caseMessage(message);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.MESSAGE_RECEIVE: {
				MessageReceive messageReceive = (MessageReceive)theEObject;
				T result = caseMessageReceive(messageReceive);
				if (result == null) result = caseStatement(messageReceive);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.MESSAGE_SEND: {
				MessageSend messageSend = (MessageSend)theEObject;
				T result = caseMessageSend(messageSend);
				if (result == null) result = caseStatement(messageSend);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.MODEL: {
				Model model = (Model)theEObject;
				T result = caseModel(model);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.OUTGOING_MESSAGE: {
				OutgoingMessage outgoingMessage = (OutgoingMessage)theEObject;
				T result = caseOutgoingMessage(outgoingMessage);
				if (result == null) result = caseMessage(outgoingMessage);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.PARALLEL_COMPOSITION: {
				ParallelComposition parallelComposition = (ParallelComposition)theEObject;
				T result = caseParallelComposition(parallelComposition);
				if (result == null) result = caseStatement(parallelComposition);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.PARAMETER: {
				Parameter parameter = (Parameter)theEObject;
				T result = caseParameter(parameter);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.PARAMETER_EXPRESSION: {
				ParameterExpression parameterExpression = (ParameterExpression)theEObject;
				T result = caseParameterExpression(parameterExpression);
				if (result == null) result = caseExpression(parameterExpression);
				if (result == null) result = caseStatement(parameterExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.PRIMITIVE_TYPE: {
				PrimitiveType primitiveType = (PrimitiveType)theEObject;
				T result = casePrimitiveType(primitiveType);
				if (result == null) result = caseType(primitiveType);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.PROCESS_CLASS: {
				ProcessClass processClass = (ProcessClass)theEObject;
				T result = caseProcessClass(processClass);
				if (result == null) result = caseClass(processClass);
				if (result == null) result = caseType(processClass);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.PROCESS_INSTANCE: {
				ProcessInstance processInstance = (ProcessInstance)theEObject;
				T result = caseProcessInstance(processInstance);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.PROCESS_METHOD: {
				ProcessMethod processMethod = (ProcessMethod)theEObject;
				T result = caseProcessMethod(processMethod);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.PROCESS_METHOD_CALL: {
				ProcessMethodCall processMethodCall = (ProcessMethodCall)theEObject;
				T result = caseProcessMethodCall(processMethodCall);
				if (result == null) result = caseStatement(processMethodCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.PORT: {
				Port port = (Port)theEObject;
				T result = casePort(port);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.REFERENCE_SELF: {
				ReferenceSelf referenceSelf = (ReferenceSelf)theEObject;
				T result = caseReferenceSelf(referenceSelf);
				if (result == null) result = caseExpression(referenceSelf);
				if (result == null) result = caseStatement(referenceSelf);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.RETURN_EXPRESSION: {
				ReturnExpression returnExpression = (ReturnExpression)theEObject;
				T result = caseReturnExpression(returnExpression);
				if (result == null) result = caseExpression(returnExpression);
				if (result == null) result = caseStatement(returnExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.SELECTION: {
				Selection selection = (Selection)theEObject;
				T result = caseSelection(selection);
				if (result == null) result = caseStatement(selection);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.SKIP: {
				Skip skip = (Skip)theEObject;
				T result = caseSkip(skip);
				if (result == null) result = caseStatement(skip);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.STATEMENT: {
				Statement statement = (Statement)theEObject;
				T result = caseStatement(statement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.STATEMENT_LIST: {
				StatementList statementList = (StatementList)theEObject;
				T result = caseStatementList(statementList);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.TIME_SYNCHRONISATION: {
				TimeSynchronisation timeSynchronisation = (TimeSynchronisation)theEObject;
				T result = caseTimeSynchronisation(timeSynchronisation);
				if (result == null) result = caseStatement(timeSynchronisation);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.TYPE: {
				Type type = (Type)theEObject;
				T result = caseType(type);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.VARIABLE: {
				Variable variable = (Variable)theEObject;
				T result = caseVariable(variable);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PooslPackage.VARIABLE_EXPRESSION: {
				VariableExpression variableExpression = (VariableExpression)theEObject;
				T result = caseVariableExpression(variableExpression);
				if (result == null) result = caseExpression(variableExpression);
				if (result == null) result = caseStatement(variableExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			default: return defaultCase(theEObject);
		}
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Abort</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Abort</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseAbort(Abort object) {
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
	public T caseAssignment(Assignment object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Atomic</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Atomic</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseAtomic(Atomic object) {
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
	public T caseChannel(Channel object) {
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
	public T caseClass(poosl.Class object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Conditional Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Conditional Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseConditionalExpression(ConditionalExpression object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Conditional Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Conditional Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseConditionalStatement(ConditionalStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseConstant(Constant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Constant Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Constant Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseConstantExpression(ConstantExpression object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Current Time</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Current Time</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCurrentTime(CurrentTime object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Data Class</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Data Class</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseDataClass(DataClass object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Data Method</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Data Method</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseDataMethod(DataMethod object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Data Method Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Data Method Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseDataMethodCall(DataMethodCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Primitive Data Method Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Primitive Data Method Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casePrimitiveDataMethodCall(PrimitiveDataMethodCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Data Object Creation</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Data Object Creation</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseDataObjectCreation(DataObjectCreation object) {
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
	public T caseExpression(Expression object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Expression List</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Expression List</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseExpressionList(ExpressionList object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Guarded Execution</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Guarded Execution</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseGuardedExecution(GuardedExecution object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Incoming Message</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Incoming Message</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseIncomingMessage(IncomingMessage object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Interrupt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Interrupt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseInterrupt(Interrupt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Loop Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Loop Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseLoopExpression(LoopExpression object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Loop Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Loop Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseLoopStatement(LoopStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Message</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Message</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseMessage(Message object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Message Receive</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Message Receive</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseMessageReceive(MessageReceive object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Message Send</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Message Send</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseMessageSend(MessageSend object) {
		return null;
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
	public T caseModel(Model object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Outgoing Message</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Outgoing Message</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseOutgoingMessage(OutgoingMessage object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Parallel Composition</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Parallel Composition</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseParallelComposition(ParallelComposition object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Parameter</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Parameter</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseParameter(Parameter object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Parameter Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Parameter Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseParameterExpression(ParameterExpression object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Primitive Type</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Primitive Type</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casePrimitiveType(PrimitiveType object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Process Class</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Process Class</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseProcessClass(ProcessClass object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Process Instance</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Process Instance</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseProcessInstance(ProcessInstance object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Process Method</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Process Method</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseProcessMethod(ProcessMethod object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Process Method Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Process Method Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseProcessMethodCall(ProcessMethodCall object) {
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
	public T casePort(Port object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Reference Self</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Reference Self</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseReferenceSelf(ReferenceSelf object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Return Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Return Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseReturnExpression(ReturnExpression object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Selection</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Selection</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSelection(Selection object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Skip</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Skip</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSkip(Skip object) {
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
	public T caseStatement(Statement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Statement List</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Statement List</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseStatementList(StatementList object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Time Synchronisation</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Time Synchronisation</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseTimeSynchronisation(TimeSynchronisation object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Type</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Type</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseType(Type object) {
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
	public T caseVariable(Variable object) {
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
	public T caseVariableExpression(VariableExpression object) {
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
	public T defaultCase(EObject object) {
		return null;
	}

} //PooslSwitch
