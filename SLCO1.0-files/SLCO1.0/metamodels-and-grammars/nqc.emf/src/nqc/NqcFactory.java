/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage
 * @generated
 */
public interface NqcFactory extends EFactory {
	/**
	 * The singleton instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	NqcFactory eINSTANCE = nqc.impl.NqcFactoryImpl.init();

	/**
	 * Returns a new object of class '<em>Acquire Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Acquire Constant</em>'.
	 * @generated
	 */
	AcquireConstant createAcquireConstant();

	/**
	 * Returns a new object of class '<em>Acquire Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Acquire Statement</em>'.
	 * @generated
	 */
	AcquireStatement createAcquireStatement();

	/**
	 * Returns a new object of class '<em>Array Expression</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Array Expression</em>'.
	 * @generated
	 */
	ArrayExpression createArrayExpression();

	/**
	 * Returns a new object of class '<em>Assignment Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Assignment Statement</em>'.
	 * @generated
	 */
	AssignmentStatement createAssignmentStatement();

	/**
	 * Returns a new object of class '<em>Binary Expression</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Binary Expression</em>'.
	 * @generated
	 */
	BinaryExpression createBinaryExpression();

	/**
	 * Returns a new object of class '<em>Binary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Binary Built In Function Call</em>'.
	 * @generated
	 */
	BinaryBuiltInFunctionCall createBinaryBuiltInFunctionCall();

	/**
	 * Returns a new object of class '<em>Binary Built In Value Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Binary Built In Value Function Call</em>'.
	 * @generated
	 */
	BinaryBuiltInValueFunctionCall createBinaryBuiltInValueFunctionCall();

	/**
	 * Returns a new object of class '<em>Block Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Block Statement</em>'.
	 * @generated
	 */
	BlockStatement createBlockStatement();

	/**
	 * Returns a new object of class '<em>Boolean Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Boolean Constant</em>'.
	 * @generated
	 */
	BooleanConstant createBooleanConstant();

	/**
	 * Returns a new object of class '<em>Break Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Break Statement</em>'.
	 * @generated
	 */
	BreakStatement createBreakStatement();

	/**
	 * Returns a new object of class '<em>Case</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Case</em>'.
	 * @generated
	 */
	Case createCase();

	/**
	 * Returns a new object of class '<em>Continue Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Continue Statement</em>'.
	 * @generated
	 */
	ContinueStatement createContinueStatement();

	/**
	 * Returns a new object of class '<em>Direction Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Direction Constant</em>'.
	 * @generated
	 */
	DirectionConstant createDirectionConstant();

	/**
	 * Returns a new object of class '<em>Display Mode Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Display Mode Constant</em>'.
	 * @generated
	 */
	DisplayModeConstant createDisplayModeConstant();

	/**
	 * Returns a new object of class '<em>Do While Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Do While Statement</em>'.
	 * @generated
	 */
	DoWhileStatement createDoWhileStatement();

	/**
	 * Returns a new object of class '<em>Empty Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Empty Statement</em>'.
	 * @generated
	 */
	EmptyStatement createEmptyStatement();

	/**
	 * Returns a new object of class '<em>Event Type Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Event Type Constant</em>'.
	 * @generated
	 */
	EventTypeConstant createEventTypeConstant();

	/**
	 * Returns a new object of class '<em>Function</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Function</em>'.
	 * @generated
	 */
	Function createFunction();

	/**
	 * Returns a new object of class '<em>Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Function Call</em>'.
	 * @generated
	 */
	FunctionCall createFunctionCall();

	/**
	 * Returns a new object of class '<em>For Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>For Statement</em>'.
	 * @generated
	 */
	ForStatement createForStatement();

	/**
	 * Returns a new object of class '<em>Go To Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Go To Statement</em>'.
	 * @generated
	 */
	GoToStatement createGoToStatement();

	/**
	 * Returns a new object of class '<em>If Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>If Statement</em>'.
	 * @generated
	 */
	IfStatement createIfStatement();

	/**
	 * Returns a new object of class '<em>Integer Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Integer Constant</em>'.
	 * @generated
	 */
	IntegerConstant createIntegerConstant();

	/**
	 * Returns a new object of class '<em>Label</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Label</em>'.
	 * @generated
	 */
	Label createLabel();

	/**
	 * Returns a new object of class '<em>Monitor Handler</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Monitor Handler</em>'.
	 * @generated
	 */
	MonitorHandler createMonitorHandler();

	/**
	 * Returns a new object of class '<em>Monitor Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Monitor Statement</em>'.
	 * @generated
	 */
	MonitorStatement createMonitorStatement();

	/**
	 * Returns a new object of class '<em>Nullary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Nullary Built In Function Call</em>'.
	 * @generated
	 */
	NullaryBuiltInFunctionCall createNullaryBuiltInFunctionCall();

	/**
	 * Returns a new object of class '<em>Nullary Built In Value Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Nullary Built In Value Function Call</em>'.
	 * @generated
	 */
	NullaryBuiltInValueFunctionCall createNullaryBuiltInValueFunctionCall();

	/**
	 * Returns a new object of class '<em>Output Mode Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Output Mode Constant</em>'.
	 * @generated
	 */
	OutputModeConstant createOutputModeConstant();

	/**
	 * Returns a new object of class '<em>Output Port Name Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Output Port Name Constant</em>'.
	 * @generated
	 */
	OutputPortNameConstant createOutputPortNameConstant();

	/**
	 * Returns a new object of class '<em>Parameter</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Parameter</em>'.
	 * @generated
	 */
	Parameter createParameter();

	/**
	 * Returns a new object of class '<em>Program</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Program</em>'.
	 * @generated
	 */
	Program createProgram();

	/**
	 * Returns a new object of class '<em>Programs</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Programs</em>'.
	 * @generated
	 */
	Programs createPrograms();

	/**
	 * Returns a new object of class '<em>Quaternary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Quaternary Built In Function Call</em>'.
	 * @generated
	 */
	QuaternaryBuiltInFunctionCall createQuaternaryBuiltInFunctionCall();

	/**
	 * Returns a new object of class '<em>Repeat Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Repeat Statement</em>'.
	 * @generated
	 */
	RepeatStatement createRepeatStatement();

	/**
	 * Returns a new object of class '<em>Return Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Return Statement</em>'.
	 * @generated
	 */
	ReturnStatement createReturnStatement();

	/**
	 * Returns a new object of class '<em>Senary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Senary Built In Function Call</em>'.
	 * @generated
	 */
	SenaryBuiltInFunctionCall createSenaryBuiltInFunctionCall();

	/**
	 * Returns a new object of class '<em>Sensor Config Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Sensor Config Constant</em>'.
	 * @generated
	 */
	SensorConfigConstant createSensorConfigConstant();

	/**
	 * Returns a new object of class '<em>Sensor Mode Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Sensor Mode Constant</em>'.
	 * @generated
	 */
	SensorModeConstant createSensorModeConstant();

	/**
	 * Returns a new object of class '<em>Sensor Name Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Sensor Name Constant</em>'.
	 * @generated
	 */
	SensorNameConstant createSensorNameConstant();

	/**
	 * Returns a new object of class '<em>Sensor Type Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Sensor Type Constant</em>'.
	 * @generated
	 */
	SensorTypeConstant createSensorTypeConstant();

	/**
	 * Returns a new object of class '<em>Serial Baud Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Serial Baud Constant</em>'.
	 * @generated
	 */
	SerialBaudConstant createSerialBaudConstant();

	/**
	 * Returns a new object of class '<em>Serial Biphase Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Serial Biphase Constant</em>'.
	 * @generated
	 */
	SerialBiphaseConstant createSerialBiphaseConstant();

	/**
	 * Returns a new object of class '<em>Serial Checksum Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Serial Checksum Constant</em>'.
	 * @generated
	 */
	SerialChecksumConstant createSerialChecksumConstant();

	/**
	 * Returns a new object of class '<em>Serial Channel Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Serial Channel Constant</em>'.
	 * @generated
	 */
	SerialChannelConstant createSerialChannelConstant();

	/**
	 * Returns a new object of class '<em>Serial Comm Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Serial Comm Constant</em>'.
	 * @generated
	 */
	SerialCommConstant createSerialCommConstant();

	/**
	 * Returns a new object of class '<em>Serial Packet Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Serial Packet Constant</em>'.
	 * @generated
	 */
	SerialPacketConstant createSerialPacketConstant();

	/**
	 * Returns a new object of class '<em>Sound Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Sound Constant</em>'.
	 * @generated
	 */
	SoundConstant createSoundConstant();

	/**
	 * Returns a new object of class '<em>Start Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Start Statement</em>'.
	 * @generated
	 */
	StartStatement createStartStatement();

	/**
	 * Returns a new object of class '<em>Stop Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Stop Statement</em>'.
	 * @generated
	 */
	StopStatement createStopStatement();

	/**
	 * Returns a new object of class '<em>Subroutine</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Subroutine</em>'.
	 * @generated
	 */
	Subroutine createSubroutine();

	/**
	 * Returns a new object of class '<em>Subroutine Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Subroutine Call</em>'.
	 * @generated
	 */
	SubroutineCall createSubroutineCall();

	/**
	 * Returns a new object of class '<em>Switch Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Switch Statement</em>'.
	 * @generated
	 */
	SwitchStatement createSwitchStatement();

	/**
	 * Returns a new object of class '<em>Task</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Task</em>'.
	 * @generated
	 */
	Task createTask();

	/**
	 * Returns a new object of class '<em>Ternary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Ternary Built In Function Call</em>'.
	 * @generated
	 */
	TernaryBuiltInFunctionCall createTernaryBuiltInFunctionCall();

	/**
	 * Returns a new object of class '<em>Ternary Expression</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Ternary Expression</em>'.
	 * @generated
	 */
	TernaryExpression createTernaryExpression();

	/**
	 * Returns a new object of class '<em>Tx Power Constant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Tx Power Constant</em>'.
	 * @generated
	 */
	TxPowerConstant createTxPowerConstant();

	/**
	 * Returns a new object of class '<em>Unary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Unary Built In Function Call</em>'.
	 * @generated
	 */
	UnaryBuiltInFunctionCall createUnaryBuiltInFunctionCall();

	/**
	 * Returns a new object of class '<em>Unary Built In Value Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Unary Built In Value Function Call</em>'.
	 * @generated
	 */
	UnaryBuiltInValueFunctionCall createUnaryBuiltInValueFunctionCall();

	/**
	 * Returns a new object of class '<em>Unary Expression</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Unary Expression</em>'.
	 * @generated
	 */
	UnaryExpression createUnaryExpression();

	/**
	 * Returns a new object of class '<em>Until Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Until Statement</em>'.
	 * @generated
	 */
	UntilStatement createUntilStatement();

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
	 * Returns a new object of class '<em>While Statement</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>While Statement</em>'.
	 * @generated
	 */
	WhileStatement createWhileStatement();

	/**
	 * Returns the package supported by this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package supported by this factory.
	 * @generated
	 */
	NqcPackage getNqcPackage();

} //NqcFactory
