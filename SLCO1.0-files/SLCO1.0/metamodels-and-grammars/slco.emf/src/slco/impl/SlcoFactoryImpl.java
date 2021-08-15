/**
 */
package slco.impl;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EDataType;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.impl.EFactoryImpl;

import org.eclipse.emf.ecore.plugin.EcorePlugin;

import slco.ArgumentType;
import slco.Assignment;
import slco.BidirectionalChannel;
import slco.BinaryOperatorExpression;
import slco.BooleanConstantExpression;
import slco.ChannelTypeEnum;
import slco.Delay;
import slco.Final;
import slco.Initial;
import slco.IntegerConstantExpression;
import slco.Model;
import slco.OperatorEnum;
import slco.Port;
import slco.PrimitiveTypeEnum;
import slco.SendSignal;
import slco.SignalArgumentExpression;
import slco.SignalArgumentVariable;
import slco.SignalReception;
import slco.SlcoFactory;
import slco.SlcoPackage;
import slco.State;
import slco.StateMachine;
import slco.StringConstantExpression;
import slco.TextualStatement;
import slco.Transition;
import slco.UnidirectionalChannel;
import slco.Variable;
import slco.VariableExpression;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Factory</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class SlcoFactoryImpl extends EFactoryImpl implements SlcoFactory {
	/**
	 * Creates the default factory implementation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SlcoFactory init() {
		try {
			SlcoFactory theSlcoFactory = (SlcoFactory)EPackage.Registry.INSTANCE.getEFactory("slco"); 
			if (theSlcoFactory != null) {
				return theSlcoFactory;
			}
		}
		catch (Exception exception) {
			EcorePlugin.INSTANCE.log(exception);
		}
		return new SlcoFactoryImpl();
	}

	/**
	 * Creates an instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SlcoFactoryImpl() {
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
			case SlcoPackage.ARGUMENT_TYPE: return createArgumentType();
			case SlcoPackage.ASSIGNMENT: return createAssignment();
			case SlcoPackage.BIDIRECTIONAL_CHANNEL: return createBidirectionalChannel();
			case SlcoPackage.BINARY_OPERATOR_EXPRESSION: return createBinaryOperatorExpression();
			case SlcoPackage.BOOLEAN_CONSTANT_EXPRESSION: return createBooleanConstantExpression();
			case SlcoPackage.CLASS: return createClass();
			case SlcoPackage.DELAY: return createDelay();
			case SlcoPackage.FINAL: return createFinal();
			case SlcoPackage.INITIAL: return createInitial();
			case SlcoPackage.INTEGER_CONSTANT_EXPRESSION: return createIntegerConstantExpression();
			case SlcoPackage.MODEL: return createModel();
			case SlcoPackage.OBJECT: return createObject();
			case SlcoPackage.PORT: return createPort();
			case SlcoPackage.SEND_SIGNAL: return createSendSignal();
			case SlcoPackage.SIGNAL_ARGUMENT_EXPRESSION: return createSignalArgumentExpression();
			case SlcoPackage.SIGNAL_ARGUMENT_VARIABLE: return createSignalArgumentVariable();
			case SlcoPackage.SIGNAL_RECEPTION: return createSignalReception();
			case SlcoPackage.STATE: return createState();
			case SlcoPackage.STATE_MACHINE: return createStateMachine();
			case SlcoPackage.STRING_CONSTANT_EXPRESSION: return createStringConstantExpression();
			case SlcoPackage.TEXTUAL_STATEMENT: return createTextualStatement();
			case SlcoPackage.TRANSITION: return createTransition();
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL: return createUnidirectionalChannel();
			case SlcoPackage.VARIABLE: return createVariable();
			case SlcoPackage.VARIABLE_EXPRESSION: return createVariableExpression();
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
			case SlcoPackage.CHANNEL_TYPE_ENUM:
				return createChannelTypeEnumFromString(eDataType, initialValue);
			case SlcoPackage.OPERATOR_ENUM:
				return createOperatorEnumFromString(eDataType, initialValue);
			case SlcoPackage.PRIMITIVE_TYPE_ENUM:
				return createPrimitiveTypeEnumFromString(eDataType, initialValue);
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
			case SlcoPackage.CHANNEL_TYPE_ENUM:
				return convertChannelTypeEnumToString(eDataType, instanceValue);
			case SlcoPackage.OPERATOR_ENUM:
				return convertOperatorEnumToString(eDataType, instanceValue);
			case SlcoPackage.PRIMITIVE_TYPE_ENUM:
				return convertPrimitiveTypeEnumToString(eDataType, instanceValue);
			default:
				throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ArgumentType createArgumentType() {
		ArgumentTypeImpl argumentType = new ArgumentTypeImpl();
		return argumentType;
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
	public BidirectionalChannel createBidirectionalChannel() {
		BidirectionalChannelImpl bidirectionalChannel = new BidirectionalChannelImpl();
		return bidirectionalChannel;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BinaryOperatorExpression createBinaryOperatorExpression() {
		BinaryOperatorExpressionImpl binaryOperatorExpression = new BinaryOperatorExpressionImpl();
		return binaryOperatorExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BooleanConstantExpression createBooleanConstantExpression() {
		BooleanConstantExpressionImpl booleanConstantExpression = new BooleanConstantExpressionImpl();
		return booleanConstantExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public slco.Class createClass() {
		ClassImpl class_ = new ClassImpl();
		return class_;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Delay createDelay() {
		DelayImpl delay = new DelayImpl();
		return delay;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Final createFinal() {
		FinalImpl final_ = new FinalImpl();
		return final_;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Initial createInitial() {
		InitialImpl initial = new InitialImpl();
		return initial;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IntegerConstantExpression createIntegerConstantExpression() {
		IntegerConstantExpressionImpl integerConstantExpression = new IntegerConstantExpressionImpl();
		return integerConstantExpression;
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
	public slco.Object createObject() {
		ObjectImpl object = new ObjectImpl();
		return object;
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
	public SendSignal createSendSignal() {
		SendSignalImpl sendSignal = new SendSignalImpl();
		return sendSignal;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SignalArgumentExpression createSignalArgumentExpression() {
		SignalArgumentExpressionImpl signalArgumentExpression = new SignalArgumentExpressionImpl();
		return signalArgumentExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SignalArgumentVariable createSignalArgumentVariable() {
		SignalArgumentVariableImpl signalArgumentVariable = new SignalArgumentVariableImpl();
		return signalArgumentVariable;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SignalReception createSignalReception() {
		SignalReceptionImpl signalReception = new SignalReceptionImpl();
		return signalReception;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public State createState() {
		StateImpl state = new StateImpl();
		return state;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public StateMachine createStateMachine() {
		StateMachineImpl stateMachine = new StateMachineImpl();
		return stateMachine;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public StringConstantExpression createStringConstantExpression() {
		StringConstantExpressionImpl stringConstantExpression = new StringConstantExpressionImpl();
		return stringConstantExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TextualStatement createTextualStatement() {
		TextualStatementImpl textualStatement = new TextualStatementImpl();
		return textualStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Transition createTransition() {
		TransitionImpl transition = new TransitionImpl();
		return transition;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UnidirectionalChannel createUnidirectionalChannel() {
		UnidirectionalChannelImpl unidirectionalChannel = new UnidirectionalChannelImpl();
		return unidirectionalChannel;
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
	public ChannelTypeEnum createChannelTypeEnumFromString(EDataType eDataType, String initialValue) {
		ChannelTypeEnum result = ChannelTypeEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertChannelTypeEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OperatorEnum createOperatorEnumFromString(EDataType eDataType, String initialValue) {
		OperatorEnum result = OperatorEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertOperatorEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
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
	public SlcoPackage getSlcoPackage() {
		return (SlcoPackage)getEPackage();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @deprecated
	 * @generated
	 */
	@Deprecated
	public static SlcoPackage getPackage() {
		return SlcoPackage.eINSTANCE;
	}

} //SlcoFactoryImpl
