/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EDataType;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.impl.EFactoryImpl;

import org.eclipse.emf.ecore.plugin.EcorePlugin;

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
import poosl.PooslFactory;
import poosl.PooslPackage;
import poosl.Port;
import poosl.PrimitiveDataMethodCall;
import poosl.PrimitiveDataMethodEnum;
import poosl.PrimitiveType;
import poosl.PrimitiveTypeEnum;
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
import poosl.Variable;
import poosl.VariableExpression;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Factory</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class PooslFactoryImpl extends EFactoryImpl implements PooslFactory {
	/**
	 * Creates the default factory implementation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static PooslFactory init() {
		try {
			PooslFactory thePooslFactory = (PooslFactory)EPackage.Registry.INSTANCE.getEFactory("poosl"); 
			if (thePooslFactory != null) {
				return thePooslFactory;
			}
		}
		catch (Exception exception) {
			EcorePlugin.INSTANCE.log(exception);
		}
		return new PooslFactoryImpl();
	}

	/**
	 * Creates an instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PooslFactoryImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EObject create(EClass eClass) {
		switch (eClass.getClassifierID()) {
			case PooslPackage.ABORT: return createAbort();
			case PooslPackage.ASSIGNMENT: return createAssignment();
			case PooslPackage.ATOMIC: return createAtomic();
			case PooslPackage.CHANNEL: return createChannel();
			case PooslPackage.CONDITIONAL_EXPRESSION: return createConditionalExpression();
			case PooslPackage.CONDITIONAL_STATEMENT: return createConditionalStatement();
			case PooslPackage.CONSTANT: return createConstant();
			case PooslPackage.CONSTANT_EXPRESSION: return createConstantExpression();
			case PooslPackage.CURRENT_TIME: return createCurrentTime();
			case PooslPackage.DATA_CLASS: return createDataClass();
			case PooslPackage.DATA_METHOD: return createDataMethod();
			case PooslPackage.DATA_METHOD_CALL: return createDataMethodCall();
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL: return createPrimitiveDataMethodCall();
			case PooslPackage.DATA_OBJECT_CREATION: return createDataObjectCreation();
			case PooslPackage.EXPRESSION_LIST: return createExpressionList();
			case PooslPackage.GUARDED_EXECUTION: return createGuardedExecution();
			case PooslPackage.INCOMING_MESSAGE: return createIncomingMessage();
			case PooslPackage.INTERRUPT: return createInterrupt();
			case PooslPackage.LOOP_EXPRESSION: return createLoopExpression();
			case PooslPackage.LOOP_STATEMENT: return createLoopStatement();
			case PooslPackage.MESSAGE: return createMessage();
			case PooslPackage.MESSAGE_RECEIVE: return createMessageReceive();
			case PooslPackage.MESSAGE_SEND: return createMessageSend();
			case PooslPackage.MODEL: return createModel();
			case PooslPackage.OUTGOING_MESSAGE: return createOutgoingMessage();
			case PooslPackage.PARALLEL_COMPOSITION: return createParallelComposition();
			case PooslPackage.PARAMETER: return createParameter();
			case PooslPackage.PARAMETER_EXPRESSION: return createParameterExpression();
			case PooslPackage.PRIMITIVE_TYPE: return createPrimitiveType();
			case PooslPackage.PROCESS_CLASS: return createProcessClass();
			case PooslPackage.PROCESS_INSTANCE: return createProcessInstance();
			case PooslPackage.PROCESS_METHOD: return createProcessMethod();
			case PooslPackage.PROCESS_METHOD_CALL: return createProcessMethodCall();
			case PooslPackage.PORT: return createPort();
			case PooslPackage.REFERENCE_SELF: return createReferenceSelf();
			case PooslPackage.RETURN_EXPRESSION: return createReturnExpression();
			case PooslPackage.SELECTION: return createSelection();
			case PooslPackage.SKIP: return createSkip();
			case PooslPackage.STATEMENT: return createStatement();
			case PooslPackage.STATEMENT_LIST: return createStatementList();
			case PooslPackage.TIME_SYNCHRONISATION: return createTimeSynchronisation();
			case PooslPackage.VARIABLE: return createVariable();
			case PooslPackage.VARIABLE_EXPRESSION: return createVariableExpression();
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
	public Object createFromString(EDataType eDataType, String initialValue) {
		switch (eDataType.getClassifierID()) {
			case PooslPackage.PRIMITIVE_TYPE_ENUM:
				return createPrimitiveTypeEnumFromString(eDataType, initialValue);
			case PooslPackage.PRIMITIVE_DATA_METHOD_ENUM:
				return createPrimitiveDataMethodEnumFromString(eDataType, initialValue);
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
	public String convertToString(EDataType eDataType, Object instanceValue) {
		switch (eDataType.getClassifierID()) {
			case PooslPackage.PRIMITIVE_TYPE_ENUM:
				return convertPrimitiveTypeEnumToString(eDataType, instanceValue);
			case PooslPackage.PRIMITIVE_DATA_METHOD_ENUM:
				return convertPrimitiveDataMethodEnumToString(eDataType, instanceValue);
			default:
				throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Abort createAbort() {
		AbortImpl abort = new AbortImpl();
		return abort;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Assignment createAssignment() {
		AssignmentImpl assignment = new AssignmentImpl();
		return assignment;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Atomic createAtomic() {
		AtomicImpl atomic = new AtomicImpl();
		return atomic;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Channel createChannel() {
		ChannelImpl channel = new ChannelImpl();
		return channel;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ConditionalExpression createConditionalExpression() {
		ConditionalExpressionImpl conditionalExpression = new ConditionalExpressionImpl();
		return conditionalExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ConditionalStatement createConditionalStatement() {
		ConditionalStatementImpl conditionalStatement = new ConditionalStatementImpl();
		return conditionalStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Constant createConstant() {
		ConstantImpl constant = new ConstantImpl();
		return constant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ConstantExpression createConstantExpression() {
		ConstantExpressionImpl constantExpression = new ConstantExpressionImpl();
		return constantExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public CurrentTime createCurrentTime() {
		CurrentTimeImpl currentTime = new CurrentTimeImpl();
		return currentTime;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DataClass createDataClass() {
		DataClassImpl dataClass = new DataClassImpl();
		return dataClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DataMethod createDataMethod() {
		DataMethodImpl dataMethod = new DataMethodImpl();
		return dataMethod;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DataMethodCall createDataMethodCall() {
		DataMethodCallImpl dataMethodCall = new DataMethodCallImpl();
		return dataMethodCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PrimitiveDataMethodCall createPrimitiveDataMethodCall() {
		PrimitiveDataMethodCallImpl primitiveDataMethodCall = new PrimitiveDataMethodCallImpl();
		return primitiveDataMethodCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DataObjectCreation createDataObjectCreation() {
		DataObjectCreationImpl dataObjectCreation = new DataObjectCreationImpl();
		return dataObjectCreation;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ExpressionList createExpressionList() {
		ExpressionListImpl expressionList = new ExpressionListImpl();
		return expressionList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public GuardedExecution createGuardedExecution() {
		GuardedExecutionImpl guardedExecution = new GuardedExecutionImpl();
		return guardedExecution;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IncomingMessage createIncomingMessage() {
		IncomingMessageImpl incomingMessage = new IncomingMessageImpl();
		return incomingMessage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Interrupt createInterrupt() {
		InterruptImpl interrupt = new InterruptImpl();
		return interrupt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public LoopExpression createLoopExpression() {
		LoopExpressionImpl loopExpression = new LoopExpressionImpl();
		return loopExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public LoopStatement createLoopStatement() {
		LoopStatementImpl loopStatement = new LoopStatementImpl();
		return loopStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Message createMessage() {
		MessageImpl message = new MessageImpl();
		return message;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MessageReceive createMessageReceive() {
		MessageReceiveImpl messageReceive = new MessageReceiveImpl();
		return messageReceive;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MessageSend createMessageSend() {
		MessageSendImpl messageSend = new MessageSendImpl();
		return messageSend;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Model createModel() {
		ModelImpl model = new ModelImpl();
		return model;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutgoingMessage createOutgoingMessage() {
		OutgoingMessageImpl outgoingMessage = new OutgoingMessageImpl();
		return outgoingMessage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ParallelComposition createParallelComposition() {
		ParallelCompositionImpl parallelComposition = new ParallelCompositionImpl();
		return parallelComposition;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Parameter createParameter() {
		ParameterImpl parameter = new ParameterImpl();
		return parameter;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ParameterExpression createParameterExpression() {
		ParameterExpressionImpl parameterExpression = new ParameterExpressionImpl();
		return parameterExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PrimitiveType createPrimitiveType() {
		PrimitiveTypeImpl primitiveType = new PrimitiveTypeImpl();
		return primitiveType;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessClass createProcessClass() {
		ProcessClassImpl processClass = new ProcessClassImpl();
		return processClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessInstance createProcessInstance() {
		ProcessInstanceImpl processInstance = new ProcessInstanceImpl();
		return processInstance;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessMethod createProcessMethod() {
		ProcessMethodImpl processMethod = new ProcessMethodImpl();
		return processMethod;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessMethodCall createProcessMethodCall() {
		ProcessMethodCallImpl processMethodCall = new ProcessMethodCallImpl();
		return processMethodCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port createPort() {
		PortImpl port = new PortImpl();
		return port;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ReferenceSelf createReferenceSelf() {
		ReferenceSelfImpl referenceSelf = new ReferenceSelfImpl();
		return referenceSelf;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ReturnExpression createReturnExpression() {
		ReturnExpressionImpl returnExpression = new ReturnExpressionImpl();
		return returnExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Selection createSelection() {
		SelectionImpl selection = new SelectionImpl();
		return selection;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Skip createSkip() {
		SkipImpl skip = new SkipImpl();
		return skip;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Statement createStatement() {
		StatementImpl statement = new StatementImpl();
		return statement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public StatementList createStatementList() {
		StatementListImpl statementList = new StatementListImpl();
		return statementList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TimeSynchronisation createTimeSynchronisation() {
		TimeSynchronisationImpl timeSynchronisation = new TimeSynchronisationImpl();
		return timeSynchronisation;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Variable createVariable() {
		VariableImpl variable = new VariableImpl();
		return variable;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public VariableExpression createVariableExpression() {
		VariableExpressionImpl variableExpression = new VariableExpressionImpl();
		return variableExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PrimitiveTypeEnum createPrimitiveTypeEnumFromString(EDataType eDataType, String initialValue) {
		PrimitiveTypeEnum result = PrimitiveTypeEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertPrimitiveTypeEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PrimitiveDataMethodEnum createPrimitiveDataMethodEnumFromString(EDataType eDataType, String initialValue) {
		PrimitiveDataMethodEnum result = PrimitiveDataMethodEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertPrimitiveDataMethodEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PooslPackage getPooslPackage() {
		return (PooslPackage)getEPackage();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @deprecated
	 * @generated
	 */
	@Deprecated
	public static PooslPackage getPackage() {
		return PooslPackage.eINSTANCE;
	}

} //PooslFactoryImpl
