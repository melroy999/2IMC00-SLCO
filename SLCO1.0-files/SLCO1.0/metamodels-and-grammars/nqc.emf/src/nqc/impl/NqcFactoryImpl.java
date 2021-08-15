/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.*;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EDataType;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.impl.EFactoryImpl;

import org.eclipse.emf.ecore.plugin.EcorePlugin;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Factory</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class NqcFactoryImpl extends EFactoryImpl implements NqcFactory {
	/**
	 * Creates the default factory implementation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static NqcFactory init() {
		try {
			NqcFactory theNqcFactory = (NqcFactory)EPackage.Registry.INSTANCE.getEFactory("nqc"); 
			if (theNqcFactory != null) {
				return theNqcFactory;
			}
		}
		catch (Exception exception) {
			EcorePlugin.INSTANCE.log(exception);
		}
		return new NqcFactoryImpl();
	}

	/**
	 * Creates an instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NqcFactoryImpl() {
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
			case NqcPackage.ACQUIRE_CONSTANT: return createAcquireConstant();
			case NqcPackage.ACQUIRE_STATEMENT: return createAcquireStatement();
			case NqcPackage.ARRAY_EXPRESSION: return createArrayExpression();
			case NqcPackage.ASSIGNMENT_STATEMENT: return createAssignmentStatement();
			case NqcPackage.BINARY_EXPRESSION: return createBinaryExpression();
			case NqcPackage.BINARY_BUILT_IN_FUNCTION_CALL: return createBinaryBuiltInFunctionCall();
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL: return createBinaryBuiltInValueFunctionCall();
			case NqcPackage.BLOCK_STATEMENT: return createBlockStatement();
			case NqcPackage.BOOLEAN_CONSTANT: return createBooleanConstant();
			case NqcPackage.BREAK_STATEMENT: return createBreakStatement();
			case NqcPackage.CASE: return createCase();
			case NqcPackage.CONTINUE_STATEMENT: return createContinueStatement();
			case NqcPackage.DIRECTION_CONSTANT: return createDirectionConstant();
			case NqcPackage.DISPLAY_MODE_CONSTANT: return createDisplayModeConstant();
			case NqcPackage.DO_WHILE_STATEMENT: return createDoWhileStatement();
			case NqcPackage.EMPTY_STATEMENT: return createEmptyStatement();
			case NqcPackage.EVENT_TYPE_CONSTANT: return createEventTypeConstant();
			case NqcPackage.FUNCTION: return createFunction();
			case NqcPackage.FUNCTION_CALL: return createFunctionCall();
			case NqcPackage.FOR_STATEMENT: return createForStatement();
			case NqcPackage.GO_TO_STATEMENT: return createGoToStatement();
			case NqcPackage.IF_STATEMENT: return createIfStatement();
			case NqcPackage.INTEGER_CONSTANT: return createIntegerConstant();
			case NqcPackage.LABEL: return createLabel();
			case NqcPackage.MONITOR_HANDLER: return createMonitorHandler();
			case NqcPackage.MONITOR_STATEMENT: return createMonitorStatement();
			case NqcPackage.NULLARY_BUILT_IN_FUNCTION_CALL: return createNullaryBuiltInFunctionCall();
			case NqcPackage.NULLARY_BUILT_IN_VALUE_FUNCTION_CALL: return createNullaryBuiltInValueFunctionCall();
			case NqcPackage.OUTPUT_MODE_CONSTANT: return createOutputModeConstant();
			case NqcPackage.OUTPUT_PORT_NAME_CONSTANT: return createOutputPortNameConstant();
			case NqcPackage.PARAMETER: return createParameter();
			case NqcPackage.PROGRAM: return createProgram();
			case NqcPackage.PROGRAMS: return createPrograms();
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL: return createQuaternaryBuiltInFunctionCall();
			case NqcPackage.REPEAT_STATEMENT: return createRepeatStatement();
			case NqcPackage.RETURN_STATEMENT: return createReturnStatement();
			case NqcPackage.SENARY_BUILT_IN_FUNCTION_CALL: return createSenaryBuiltInFunctionCall();
			case NqcPackage.SENSOR_CONFIG_CONSTANT: return createSensorConfigConstant();
			case NqcPackage.SENSOR_MODE_CONSTANT: return createSensorModeConstant();
			case NqcPackage.SENSOR_NAME_CONSTANT: return createSensorNameConstant();
			case NqcPackage.SENSOR_TYPE_CONSTANT: return createSensorTypeConstant();
			case NqcPackage.SERIAL_BAUD_CONSTANT: return createSerialBaudConstant();
			case NqcPackage.SERIAL_BIPHASE_CONSTANT: return createSerialBiphaseConstant();
			case NqcPackage.SERIAL_CHECKSUM_CONSTANT: return createSerialChecksumConstant();
			case NqcPackage.SERIAL_CHANNEL_CONSTANT: return createSerialChannelConstant();
			case NqcPackage.SERIAL_COMM_CONSTANT: return createSerialCommConstant();
			case NqcPackage.SERIAL_PACKET_CONSTANT: return createSerialPacketConstant();
			case NqcPackage.SOUND_CONSTANT: return createSoundConstant();
			case NqcPackage.START_STATEMENT: return createStartStatement();
			case NqcPackage.STOP_STATEMENT: return createStopStatement();
			case NqcPackage.SUBROUTINE: return createSubroutine();
			case NqcPackage.SUBROUTINE_CALL: return createSubroutineCall();
			case NqcPackage.SWITCH_STATEMENT: return createSwitchStatement();
			case NqcPackage.TASK: return createTask();
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL: return createTernaryBuiltInFunctionCall();
			case NqcPackage.TERNARY_EXPRESSION: return createTernaryExpression();
			case NqcPackage.TX_POWER_CONSTANT: return createTxPowerConstant();
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL: return createUnaryBuiltInFunctionCall();
			case NqcPackage.UNARY_BUILT_IN_VALUE_FUNCTION_CALL: return createUnaryBuiltInValueFunctionCall();
			case NqcPackage.UNARY_EXPRESSION: return createUnaryExpression();
			case NqcPackage.UNTIL_STATEMENT: return createUntilStatement();
			case NqcPackage.VARIABLE: return createVariable();
			case NqcPackage.VARIABLE_EXPRESSION: return createVariableExpression();
			case NqcPackage.WHILE_STATEMENT: return createWhileStatement();
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
			case NqcPackage.ACQUIRE_ENUM:
				return createAcquireEnumFromString(eDataType, initialValue);
			case NqcPackage.ASSIGNMENT_STATEMENT_ENUM:
				return createAssignmentStatementEnumFromString(eDataType, initialValue);
			case NqcPackage.BINARY_OPERATOR_ENUM:
				return createBinaryOperatorEnumFromString(eDataType, initialValue);
			case NqcPackage.BUILT_IN_BINARY_FUNCTION_ENUM:
				return createBuiltInBinaryFunctionEnumFromString(eDataType, initialValue);
			case NqcPackage.BUILT_IN_BINARY_VALUE_FUNCTION_ENUM:
				return createBuiltInBinaryValueFunctionEnumFromString(eDataType, initialValue);
			case NqcPackage.BUILT_IN_NULLARY_FUNCTION_ENUM:
				return createBuiltInNullaryFunctionEnumFromString(eDataType, initialValue);
			case NqcPackage.BUILT_IN_NULLARY_VALUE_FUNCTION_ENUM:
				return createBuiltInNullaryValueFunctionEnumFromString(eDataType, initialValue);
			case NqcPackage.BUILT_IN_UNARY_VALUE_FUNCTION_ENUM:
				return createBuiltInUnaryValueFunctionEnumFromString(eDataType, initialValue);
			case NqcPackage.BUILT_IN_UNARY_FUNCTION_ENUM:
				return createBuiltInUnaryFunctionEnumFromString(eDataType, initialValue);
			case NqcPackage.BUILT_IN_QUATERNARY_FUNCTION_ENUM:
				return createBuiltInQuaternaryFunctionEnumFromString(eDataType, initialValue);
			case NqcPackage.BUILT_IN_SENARY_FUNCTION_ENUM:
				return createBuiltInSenaryFunctionEnumFromString(eDataType, initialValue);
			case NqcPackage.BUILT_IN_TERNARY_FUNCTION_ENUM:
				return createBuiltInTernaryFunctionEnumFromString(eDataType, initialValue);
			case NqcPackage.DIRECTION_ENUM:
				return createDirectionEnumFromString(eDataType, initialValue);
			case NqcPackage.DISPLAY_MODE_ENUM:
				return createDisplayModeEnumFromString(eDataType, initialValue);
			case NqcPackage.EVENT_TYPE_ENUM:
				return createEventTypeEnumFromString(eDataType, initialValue);
			case NqcPackage.OUTPUT_MODE_ENUM:
				return createOutputModeEnumFromString(eDataType, initialValue);
			case NqcPackage.OUTPUT_PORT_NAME_ENUM:
				return createOutputPortNameEnumFromString(eDataType, initialValue);
			case NqcPackage.SERIAL_BAUD_ENUM:
				return createSerialBaudEnumFromString(eDataType, initialValue);
			case NqcPackage.SERIAL_BIPHASE_ENUM:
				return createSerialBiphaseEnumFromString(eDataType, initialValue);
			case NqcPackage.SERIAL_CHANNEL_ENUM:
				return createSerialChannelEnumFromString(eDataType, initialValue);
			case NqcPackage.SERIAL_CHECKSUM_ENUM:
				return createSerialChecksumEnumFromString(eDataType, initialValue);
			case NqcPackage.SERIAL_COMM_ENUM:
				return createSerialCommEnumFromString(eDataType, initialValue);
			case NqcPackage.SERIAL_PACKET_ENUM:
				return createSerialPacketEnumFromString(eDataType, initialValue);
			case NqcPackage.SENSOR_CONFIG_ENUM:
				return createSensorConfigEnumFromString(eDataType, initialValue);
			case NqcPackage.SENSOR_MODE_ENUM:
				return createSensorModeEnumFromString(eDataType, initialValue);
			case NqcPackage.SENSOR_NAME_ENUM:
				return createSensorNameEnumFromString(eDataType, initialValue);
			case NqcPackage.SENSOR_TYPE_ENUM:
				return createSensorTypeEnumFromString(eDataType, initialValue);
			case NqcPackage.SOUND_ENUM:
				return createSoundEnumFromString(eDataType, initialValue);
			case NqcPackage.TERNARY_OPERATOR_ENUM:
				return createTernaryOperatorEnumFromString(eDataType, initialValue);
			case NqcPackage.TX_POWER_ENUM:
				return createTxPowerEnumFromString(eDataType, initialValue);
			case NqcPackage.TYPE_ENUM:
				return createTypeEnumFromString(eDataType, initialValue);
			case NqcPackage.UNARY_OPERATOR_ENUM:
				return createUnaryOperatorEnumFromString(eDataType, initialValue);
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
			case NqcPackage.ACQUIRE_ENUM:
				return convertAcquireEnumToString(eDataType, instanceValue);
			case NqcPackage.ASSIGNMENT_STATEMENT_ENUM:
				return convertAssignmentStatementEnumToString(eDataType, instanceValue);
			case NqcPackage.BINARY_OPERATOR_ENUM:
				return convertBinaryOperatorEnumToString(eDataType, instanceValue);
			case NqcPackage.BUILT_IN_BINARY_FUNCTION_ENUM:
				return convertBuiltInBinaryFunctionEnumToString(eDataType, instanceValue);
			case NqcPackage.BUILT_IN_BINARY_VALUE_FUNCTION_ENUM:
				return convertBuiltInBinaryValueFunctionEnumToString(eDataType, instanceValue);
			case NqcPackage.BUILT_IN_NULLARY_FUNCTION_ENUM:
				return convertBuiltInNullaryFunctionEnumToString(eDataType, instanceValue);
			case NqcPackage.BUILT_IN_NULLARY_VALUE_FUNCTION_ENUM:
				return convertBuiltInNullaryValueFunctionEnumToString(eDataType, instanceValue);
			case NqcPackage.BUILT_IN_UNARY_VALUE_FUNCTION_ENUM:
				return convertBuiltInUnaryValueFunctionEnumToString(eDataType, instanceValue);
			case NqcPackage.BUILT_IN_UNARY_FUNCTION_ENUM:
				return convertBuiltInUnaryFunctionEnumToString(eDataType, instanceValue);
			case NqcPackage.BUILT_IN_QUATERNARY_FUNCTION_ENUM:
				return convertBuiltInQuaternaryFunctionEnumToString(eDataType, instanceValue);
			case NqcPackage.BUILT_IN_SENARY_FUNCTION_ENUM:
				return convertBuiltInSenaryFunctionEnumToString(eDataType, instanceValue);
			case NqcPackage.BUILT_IN_TERNARY_FUNCTION_ENUM:
				return convertBuiltInTernaryFunctionEnumToString(eDataType, instanceValue);
			case NqcPackage.DIRECTION_ENUM:
				return convertDirectionEnumToString(eDataType, instanceValue);
			case NqcPackage.DISPLAY_MODE_ENUM:
				return convertDisplayModeEnumToString(eDataType, instanceValue);
			case NqcPackage.EVENT_TYPE_ENUM:
				return convertEventTypeEnumToString(eDataType, instanceValue);
			case NqcPackage.OUTPUT_MODE_ENUM:
				return convertOutputModeEnumToString(eDataType, instanceValue);
			case NqcPackage.OUTPUT_PORT_NAME_ENUM:
				return convertOutputPortNameEnumToString(eDataType, instanceValue);
			case NqcPackage.SERIAL_BAUD_ENUM:
				return convertSerialBaudEnumToString(eDataType, instanceValue);
			case NqcPackage.SERIAL_BIPHASE_ENUM:
				return convertSerialBiphaseEnumToString(eDataType, instanceValue);
			case NqcPackage.SERIAL_CHANNEL_ENUM:
				return convertSerialChannelEnumToString(eDataType, instanceValue);
			case NqcPackage.SERIAL_CHECKSUM_ENUM:
				return convertSerialChecksumEnumToString(eDataType, instanceValue);
			case NqcPackage.SERIAL_COMM_ENUM:
				return convertSerialCommEnumToString(eDataType, instanceValue);
			case NqcPackage.SERIAL_PACKET_ENUM:
				return convertSerialPacketEnumToString(eDataType, instanceValue);
			case NqcPackage.SENSOR_CONFIG_ENUM:
				return convertSensorConfigEnumToString(eDataType, instanceValue);
			case NqcPackage.SENSOR_MODE_ENUM:
				return convertSensorModeEnumToString(eDataType, instanceValue);
			case NqcPackage.SENSOR_NAME_ENUM:
				return convertSensorNameEnumToString(eDataType, instanceValue);
			case NqcPackage.SENSOR_TYPE_ENUM:
				return convertSensorTypeEnumToString(eDataType, instanceValue);
			case NqcPackage.SOUND_ENUM:
				return convertSoundEnumToString(eDataType, instanceValue);
			case NqcPackage.TERNARY_OPERATOR_ENUM:
				return convertTernaryOperatorEnumToString(eDataType, instanceValue);
			case NqcPackage.TX_POWER_ENUM:
				return convertTxPowerEnumToString(eDataType, instanceValue);
			case NqcPackage.TYPE_ENUM:
				return convertTypeEnumToString(eDataType, instanceValue);
			case NqcPackage.UNARY_OPERATOR_ENUM:
				return convertUnaryOperatorEnumToString(eDataType, instanceValue);
			default:
				throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AcquireConstant createAcquireConstant() {
		AcquireConstantImpl acquireConstant = new AcquireConstantImpl();
		return acquireConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AcquireStatement createAcquireStatement() {
		AcquireStatementImpl acquireStatement = new AcquireStatementImpl();
		return acquireStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ArrayExpression createArrayExpression() {
		ArrayExpressionImpl arrayExpression = new ArrayExpressionImpl();
		return arrayExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AssignmentStatement createAssignmentStatement() {
		AssignmentStatementImpl assignmentStatement = new AssignmentStatementImpl();
		return assignmentStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BinaryExpression createBinaryExpression() {
		BinaryExpressionImpl binaryExpression = new BinaryExpressionImpl();
		return binaryExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BinaryBuiltInFunctionCall createBinaryBuiltInFunctionCall() {
		BinaryBuiltInFunctionCallImpl binaryBuiltInFunctionCall = new BinaryBuiltInFunctionCallImpl();
		return binaryBuiltInFunctionCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BinaryBuiltInValueFunctionCall createBinaryBuiltInValueFunctionCall() {
		BinaryBuiltInValueFunctionCallImpl binaryBuiltInValueFunctionCall = new BinaryBuiltInValueFunctionCallImpl();
		return binaryBuiltInValueFunctionCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BlockStatement createBlockStatement() {
		BlockStatementImpl blockStatement = new BlockStatementImpl();
		return blockStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BooleanConstant createBooleanConstant() {
		BooleanConstantImpl booleanConstant = new BooleanConstantImpl();
		return booleanConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BreakStatement createBreakStatement() {
		BreakStatementImpl breakStatement = new BreakStatementImpl();
		return breakStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Case createCase() {
		CaseImpl case_ = new CaseImpl();
		return case_;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ContinueStatement createContinueStatement() {
		ContinueStatementImpl continueStatement = new ContinueStatementImpl();
		return continueStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DirectionConstant createDirectionConstant() {
		DirectionConstantImpl directionConstant = new DirectionConstantImpl();
		return directionConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DisplayModeConstant createDisplayModeConstant() {
		DisplayModeConstantImpl displayModeConstant = new DisplayModeConstantImpl();
		return displayModeConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DoWhileStatement createDoWhileStatement() {
		DoWhileStatementImpl doWhileStatement = new DoWhileStatementImpl();
		return doWhileStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EmptyStatement createEmptyStatement() {
		EmptyStatementImpl emptyStatement = new EmptyStatementImpl();
		return emptyStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EventTypeConstant createEventTypeConstant() {
		EventTypeConstantImpl eventTypeConstant = new EventTypeConstantImpl();
		return eventTypeConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Function createFunction() {
		FunctionImpl function = new FunctionImpl();
		return function;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public FunctionCall createFunctionCall() {
		FunctionCallImpl functionCall = new FunctionCallImpl();
		return functionCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ForStatement createForStatement() {
		ForStatementImpl forStatement = new ForStatementImpl();
		return forStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public GoToStatement createGoToStatement() {
		GoToStatementImpl goToStatement = new GoToStatementImpl();
		return goToStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IfStatement createIfStatement() {
		IfStatementImpl ifStatement = new IfStatementImpl();
		return ifStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IntegerConstant createIntegerConstant() {
		IntegerConstantImpl integerConstant = new IntegerConstantImpl();
		return integerConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Label createLabel() {
		LabelImpl label = new LabelImpl();
		return label;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MonitorHandler createMonitorHandler() {
		MonitorHandlerImpl monitorHandler = new MonitorHandlerImpl();
		return monitorHandler;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MonitorStatement createMonitorStatement() {
		MonitorStatementImpl monitorStatement = new MonitorStatementImpl();
		return monitorStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NullaryBuiltInFunctionCall createNullaryBuiltInFunctionCall() {
		NullaryBuiltInFunctionCallImpl nullaryBuiltInFunctionCall = new NullaryBuiltInFunctionCallImpl();
		return nullaryBuiltInFunctionCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NullaryBuiltInValueFunctionCall createNullaryBuiltInValueFunctionCall() {
		NullaryBuiltInValueFunctionCallImpl nullaryBuiltInValueFunctionCall = new NullaryBuiltInValueFunctionCallImpl();
		return nullaryBuiltInValueFunctionCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutputModeConstant createOutputModeConstant() {
		OutputModeConstantImpl outputModeConstant = new OutputModeConstantImpl();
		return outputModeConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutputPortNameConstant createOutputPortNameConstant() {
		OutputPortNameConstantImpl outputPortNameConstant = new OutputPortNameConstantImpl();
		return outputPortNameConstant;
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
	public Program createProgram() {
		ProgramImpl program = new ProgramImpl();
		return program;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Programs createPrograms() {
		ProgramsImpl programs = new ProgramsImpl();
		return programs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public QuaternaryBuiltInFunctionCall createQuaternaryBuiltInFunctionCall() {
		QuaternaryBuiltInFunctionCallImpl quaternaryBuiltInFunctionCall = new QuaternaryBuiltInFunctionCallImpl();
		return quaternaryBuiltInFunctionCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public RepeatStatement createRepeatStatement() {
		RepeatStatementImpl repeatStatement = new RepeatStatementImpl();
		return repeatStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ReturnStatement createReturnStatement() {
		ReturnStatementImpl returnStatement = new ReturnStatementImpl();
		return returnStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SenaryBuiltInFunctionCall createSenaryBuiltInFunctionCall() {
		SenaryBuiltInFunctionCallImpl senaryBuiltInFunctionCall = new SenaryBuiltInFunctionCallImpl();
		return senaryBuiltInFunctionCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorConfigConstant createSensorConfigConstant() {
		SensorConfigConstantImpl sensorConfigConstant = new SensorConfigConstantImpl();
		return sensorConfigConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorModeConstant createSensorModeConstant() {
		SensorModeConstantImpl sensorModeConstant = new SensorModeConstantImpl();
		return sensorModeConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorNameConstant createSensorNameConstant() {
		SensorNameConstantImpl sensorNameConstant = new SensorNameConstantImpl();
		return sensorNameConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorTypeConstant createSensorTypeConstant() {
		SensorTypeConstantImpl sensorTypeConstant = new SensorTypeConstantImpl();
		return sensorTypeConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialBaudConstant createSerialBaudConstant() {
		SerialBaudConstantImpl serialBaudConstant = new SerialBaudConstantImpl();
		return serialBaudConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialBiphaseConstant createSerialBiphaseConstant() {
		SerialBiphaseConstantImpl serialBiphaseConstant = new SerialBiphaseConstantImpl();
		return serialBiphaseConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialChecksumConstant createSerialChecksumConstant() {
		SerialChecksumConstantImpl serialChecksumConstant = new SerialChecksumConstantImpl();
		return serialChecksumConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialChannelConstant createSerialChannelConstant() {
		SerialChannelConstantImpl serialChannelConstant = new SerialChannelConstantImpl();
		return serialChannelConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialCommConstant createSerialCommConstant() {
		SerialCommConstantImpl serialCommConstant = new SerialCommConstantImpl();
		return serialCommConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialPacketConstant createSerialPacketConstant() {
		SerialPacketConstantImpl serialPacketConstant = new SerialPacketConstantImpl();
		return serialPacketConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SoundConstant createSoundConstant() {
		SoundConstantImpl soundConstant = new SoundConstantImpl();
		return soundConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public StartStatement createStartStatement() {
		StartStatementImpl startStatement = new StartStatementImpl();
		return startStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public StopStatement createStopStatement() {
		StopStatementImpl stopStatement = new StopStatementImpl();
		return stopStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Subroutine createSubroutine() {
		SubroutineImpl subroutine = new SubroutineImpl();
		return subroutine;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SubroutineCall createSubroutineCall() {
		SubroutineCallImpl subroutineCall = new SubroutineCallImpl();
		return subroutineCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SwitchStatement createSwitchStatement() {
		SwitchStatementImpl switchStatement = new SwitchStatementImpl();
		return switchStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Task createTask() {
		TaskImpl task = new TaskImpl();
		return task;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TernaryBuiltInFunctionCall createTernaryBuiltInFunctionCall() {
		TernaryBuiltInFunctionCallImpl ternaryBuiltInFunctionCall = new TernaryBuiltInFunctionCallImpl();
		return ternaryBuiltInFunctionCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TernaryExpression createTernaryExpression() {
		TernaryExpressionImpl ternaryExpression = new TernaryExpressionImpl();
		return ternaryExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TxPowerConstant createTxPowerConstant() {
		TxPowerConstantImpl txPowerConstant = new TxPowerConstantImpl();
		return txPowerConstant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UnaryBuiltInFunctionCall createUnaryBuiltInFunctionCall() {
		UnaryBuiltInFunctionCallImpl unaryBuiltInFunctionCall = new UnaryBuiltInFunctionCallImpl();
		return unaryBuiltInFunctionCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UnaryBuiltInValueFunctionCall createUnaryBuiltInValueFunctionCall() {
		UnaryBuiltInValueFunctionCallImpl unaryBuiltInValueFunctionCall = new UnaryBuiltInValueFunctionCallImpl();
		return unaryBuiltInValueFunctionCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UnaryExpression createUnaryExpression() {
		UnaryExpressionImpl unaryExpression = new UnaryExpressionImpl();
		return unaryExpression;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UntilStatement createUntilStatement() {
		UntilStatementImpl untilStatement = new UntilStatementImpl();
		return untilStatement;
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
	public WhileStatement createWhileStatement() {
		WhileStatementImpl whileStatement = new WhileStatementImpl();
		return whileStatement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AcquireEnum createAcquireEnumFromString(EDataType eDataType, String initialValue) {
		AcquireEnum result = AcquireEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertAcquireEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AssignmentStatementEnum createAssignmentStatementEnumFromString(EDataType eDataType, String initialValue) {
		AssignmentStatementEnum result = AssignmentStatementEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertAssignmentStatementEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BinaryOperatorEnum createBinaryOperatorEnumFromString(EDataType eDataType, String initialValue) {
		BinaryOperatorEnum result = BinaryOperatorEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertBinaryOperatorEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInBinaryFunctionEnum createBuiltInBinaryFunctionEnumFromString(EDataType eDataType, String initialValue) {
		BuiltInBinaryFunctionEnum result = BuiltInBinaryFunctionEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertBuiltInBinaryFunctionEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInBinaryValueFunctionEnum createBuiltInBinaryValueFunctionEnumFromString(EDataType eDataType, String initialValue) {
		BuiltInBinaryValueFunctionEnum result = BuiltInBinaryValueFunctionEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertBuiltInBinaryValueFunctionEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInNullaryFunctionEnum createBuiltInNullaryFunctionEnumFromString(EDataType eDataType, String initialValue) {
		BuiltInNullaryFunctionEnum result = BuiltInNullaryFunctionEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertBuiltInNullaryFunctionEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInNullaryValueFunctionEnum createBuiltInNullaryValueFunctionEnumFromString(EDataType eDataType, String initialValue) {
		BuiltInNullaryValueFunctionEnum result = BuiltInNullaryValueFunctionEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertBuiltInNullaryValueFunctionEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInUnaryValueFunctionEnum createBuiltInUnaryValueFunctionEnumFromString(EDataType eDataType, String initialValue) {
		BuiltInUnaryValueFunctionEnum result = BuiltInUnaryValueFunctionEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertBuiltInUnaryValueFunctionEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInUnaryFunctionEnum createBuiltInUnaryFunctionEnumFromString(EDataType eDataType, String initialValue) {
		BuiltInUnaryFunctionEnum result = BuiltInUnaryFunctionEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertBuiltInUnaryFunctionEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInQuaternaryFunctionEnum createBuiltInQuaternaryFunctionEnumFromString(EDataType eDataType, String initialValue) {
		BuiltInQuaternaryFunctionEnum result = BuiltInQuaternaryFunctionEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertBuiltInQuaternaryFunctionEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInSenaryFunctionEnum createBuiltInSenaryFunctionEnumFromString(EDataType eDataType, String initialValue) {
		BuiltInSenaryFunctionEnum result = BuiltInSenaryFunctionEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertBuiltInSenaryFunctionEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInTernaryFunctionEnum createBuiltInTernaryFunctionEnumFromString(EDataType eDataType, String initialValue) {
		BuiltInTernaryFunctionEnum result = BuiltInTernaryFunctionEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertBuiltInTernaryFunctionEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DirectionEnum createDirectionEnumFromString(EDataType eDataType, String initialValue) {
		DirectionEnum result = DirectionEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertDirectionEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DisplayModeEnum createDisplayModeEnumFromString(EDataType eDataType, String initialValue) {
		DisplayModeEnum result = DisplayModeEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertDisplayModeEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EventTypeEnum createEventTypeEnumFromString(EDataType eDataType, String initialValue) {
		EventTypeEnum result = EventTypeEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertEventTypeEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutputModeEnum createOutputModeEnumFromString(EDataType eDataType, String initialValue) {
		OutputModeEnum result = OutputModeEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertOutputModeEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutputPortNameEnum createOutputPortNameEnumFromString(EDataType eDataType, String initialValue) {
		OutputPortNameEnum result = OutputPortNameEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertOutputPortNameEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialBaudEnum createSerialBaudEnumFromString(EDataType eDataType, String initialValue) {
		SerialBaudEnum result = SerialBaudEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSerialBaudEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialBiphaseEnum createSerialBiphaseEnumFromString(EDataType eDataType, String initialValue) {
		SerialBiphaseEnum result = SerialBiphaseEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSerialBiphaseEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialChannelEnum createSerialChannelEnumFromString(EDataType eDataType, String initialValue) {
		SerialChannelEnum result = SerialChannelEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSerialChannelEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialChecksumEnum createSerialChecksumEnumFromString(EDataType eDataType, String initialValue) {
		SerialChecksumEnum result = SerialChecksumEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSerialChecksumEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialCommEnum createSerialCommEnumFromString(EDataType eDataType, String initialValue) {
		SerialCommEnum result = SerialCommEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSerialCommEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialPacketEnum createSerialPacketEnumFromString(EDataType eDataType, String initialValue) {
		SerialPacketEnum result = SerialPacketEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSerialPacketEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorConfigEnum createSensorConfigEnumFromString(EDataType eDataType, String initialValue) {
		SensorConfigEnum result = SensorConfigEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSensorConfigEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorModeEnum createSensorModeEnumFromString(EDataType eDataType, String initialValue) {
		SensorModeEnum result = SensorModeEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSensorModeEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorNameEnum createSensorNameEnumFromString(EDataType eDataType, String initialValue) {
		SensorNameEnum result = SensorNameEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSensorNameEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorTypeEnum createSensorTypeEnumFromString(EDataType eDataType, String initialValue) {
		SensorTypeEnum result = SensorTypeEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSensorTypeEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SoundEnum createSoundEnumFromString(EDataType eDataType, String initialValue) {
		SoundEnum result = SoundEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertSoundEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TernaryOperatorEnum createTernaryOperatorEnumFromString(EDataType eDataType, String initialValue) {
		TernaryOperatorEnum result = TernaryOperatorEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertTernaryOperatorEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TxPowerEnum createTxPowerEnumFromString(EDataType eDataType, String initialValue) {
		TxPowerEnum result = TxPowerEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertTxPowerEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TypeEnum createTypeEnumFromString(EDataType eDataType, String initialValue) {
		TypeEnum result = TypeEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertTypeEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UnaryOperatorEnum createUnaryOperatorEnumFromString(EDataType eDataType, String initialValue) {
		UnaryOperatorEnum result = UnaryOperatorEnum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertUnaryOperatorEnumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NqcPackage getNqcPackage() {
		return (NqcPackage)getEPackage();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @deprecated
	 * @generated
	 */
	@Deprecated
	public static NqcPackage getPackage() {
		return NqcPackage.eINSTANCE;
	}

} //NqcFactoryImpl
