/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.util;

import java.util.List;

import nqc.*;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.util.Switch;

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
 * @see nqc.NqcPackage
 * @generated
 */
public class NqcSwitch<T> extends Switch<T> {
	/**
	 * The cached model package
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static NqcPackage modelPackage;

	/**
	 * Creates an instance of the switch.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NqcSwitch() {
		if (modelPackage == null) {
			modelPackage = NqcPackage.eINSTANCE;
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
			case NqcPackage.ACQUIRE_CONSTANT: {
				AcquireConstant acquireConstant = (AcquireConstant)theEObject;
				T result = caseAcquireConstant(acquireConstant);
				if (result == null) result = caseConstantExpression(acquireConstant);
				if (result == null) result = caseValueExpression(acquireConstant);
				if (result == null) result = caseExpression(acquireConstant);
				if (result == null) result = caseStatement(acquireConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.ACQUIRE_STATEMENT: {
				AcquireStatement acquireStatement = (AcquireStatement)theEObject;
				T result = caseAcquireStatement(acquireStatement);
				if (result == null) result = caseStatement(acquireStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.ARRAY_EXPRESSION: {
				ArrayExpression arrayExpression = (ArrayExpression)theEObject;
				T result = caseArrayExpression(arrayExpression);
				if (result == null) result = caseVariableExpression(arrayExpression);
				if (result == null) result = caseValueExpression(arrayExpression);
				if (result == null) result = caseExpression(arrayExpression);
				if (result == null) result = caseStatement(arrayExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.ASSIGNMENT_STATEMENT: {
				AssignmentStatement assignmentStatement = (AssignmentStatement)theEObject;
				T result = caseAssignmentStatement(assignmentStatement);
				if (result == null) result = caseStatement(assignmentStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.BINARY_EXPRESSION: {
				BinaryExpression binaryExpression = (BinaryExpression)theEObject;
				T result = caseBinaryExpression(binaryExpression);
				if (result == null) result = caseCompoundExpression(binaryExpression);
				if (result == null) result = caseExpression(binaryExpression);
				if (result == null) result = caseStatement(binaryExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.BINARY_BUILT_IN_FUNCTION_CALL: {
				BinaryBuiltInFunctionCall binaryBuiltInFunctionCall = (BinaryBuiltInFunctionCall)theEObject;
				T result = caseBinaryBuiltInFunctionCall(binaryBuiltInFunctionCall);
				if (result == null) result = caseBuiltInFunctionCall(binaryBuiltInFunctionCall);
				if (result == null) result = caseCallStatement(binaryBuiltInFunctionCall);
				if (result == null) result = caseStatement(binaryBuiltInFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL: {
				BinaryBuiltInValueFunctionCall binaryBuiltInValueFunctionCall = (BinaryBuiltInValueFunctionCall)theEObject;
				T result = caseBinaryBuiltInValueFunctionCall(binaryBuiltInValueFunctionCall);
				if (result == null) result = caseBuiltInValueFunctionCall(binaryBuiltInValueFunctionCall);
				if (result == null) result = caseExpression(binaryBuiltInValueFunctionCall);
				if (result == null) result = caseStatement(binaryBuiltInValueFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.BLOCK_STATEMENT: {
				BlockStatement blockStatement = (BlockStatement)theEObject;
				T result = caseBlockStatement(blockStatement);
				if (result == null) result = caseStatement(blockStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.BOOLEAN_CONSTANT: {
				BooleanConstant booleanConstant = (BooleanConstant)theEObject;
				T result = caseBooleanConstant(booleanConstant);
				if (result == null) result = caseConstantExpression(booleanConstant);
				if (result == null) result = caseValueExpression(booleanConstant);
				if (result == null) result = caseExpression(booleanConstant);
				if (result == null) result = caseStatement(booleanConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.BREAK_STATEMENT: {
				BreakStatement breakStatement = (BreakStatement)theEObject;
				T result = caseBreakStatement(breakStatement);
				if (result == null) result = caseStatement(breakStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.BUILT_IN_FUNCTION_CALL: {
				BuiltInFunctionCall builtInFunctionCall = (BuiltInFunctionCall)theEObject;
				T result = caseBuiltInFunctionCall(builtInFunctionCall);
				if (result == null) result = caseCallStatement(builtInFunctionCall);
				if (result == null) result = caseStatement(builtInFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.BUILT_IN_VALUE_FUNCTION_CALL: {
				BuiltInValueFunctionCall builtInValueFunctionCall = (BuiltInValueFunctionCall)theEObject;
				T result = caseBuiltInValueFunctionCall(builtInValueFunctionCall);
				if (result == null) result = caseExpression(builtInValueFunctionCall);
				if (result == null) result = caseStatement(builtInValueFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.CALL_STATEMENT: {
				CallStatement callStatement = (CallStatement)theEObject;
				T result = caseCallStatement(callStatement);
				if (result == null) result = caseStatement(callStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.CASE: {
				Case case_ = (Case)theEObject;
				T result = caseCase(case_);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.COMPOUND_EXPRESSION: {
				CompoundExpression compoundExpression = (CompoundExpression)theEObject;
				T result = caseCompoundExpression(compoundExpression);
				if (result == null) result = caseExpression(compoundExpression);
				if (result == null) result = caseStatement(compoundExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.CONSTANT_EXPRESSION: {
				ConstantExpression constantExpression = (ConstantExpression)theEObject;
				T result = caseConstantExpression(constantExpression);
				if (result == null) result = caseValueExpression(constantExpression);
				if (result == null) result = caseExpression(constantExpression);
				if (result == null) result = caseStatement(constantExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.CONTINUE_STATEMENT: {
				ContinueStatement continueStatement = (ContinueStatement)theEObject;
				T result = caseContinueStatement(continueStatement);
				if (result == null) result = caseStatement(continueStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.CONTROL_STRUCTURE: {
				ControlStructure controlStructure = (ControlStructure)theEObject;
				T result = caseControlStructure(controlStructure);
				if (result == null) result = caseStatement(controlStructure);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.DIRECTION_CONSTANT: {
				DirectionConstant directionConstant = (DirectionConstant)theEObject;
				T result = caseDirectionConstant(directionConstant);
				if (result == null) result = caseConstantExpression(directionConstant);
				if (result == null) result = caseValueExpression(directionConstant);
				if (result == null) result = caseExpression(directionConstant);
				if (result == null) result = caseStatement(directionConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.DISPLAY_MODE_CONSTANT: {
				DisplayModeConstant displayModeConstant = (DisplayModeConstant)theEObject;
				T result = caseDisplayModeConstant(displayModeConstant);
				if (result == null) result = caseConstantExpression(displayModeConstant);
				if (result == null) result = caseValueExpression(displayModeConstant);
				if (result == null) result = caseExpression(displayModeConstant);
				if (result == null) result = caseStatement(displayModeConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.DO_WHILE_STATEMENT: {
				DoWhileStatement doWhileStatement = (DoWhileStatement)theEObject;
				T result = caseDoWhileStatement(doWhileStatement);
				if (result == null) result = caseControlStructure(doWhileStatement);
				if (result == null) result = caseStatement(doWhileStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.EMPTY_STATEMENT: {
				EmptyStatement emptyStatement = (EmptyStatement)theEObject;
				T result = caseEmptyStatement(emptyStatement);
				if (result == null) result = caseStatement(emptyStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.EVENT_TYPE_CONSTANT: {
				EventTypeConstant eventTypeConstant = (EventTypeConstant)theEObject;
				T result = caseEventTypeConstant(eventTypeConstant);
				if (result == null) result = caseConstantExpression(eventTypeConstant);
				if (result == null) result = caseValueExpression(eventTypeConstant);
				if (result == null) result = caseExpression(eventTypeConstant);
				if (result == null) result = caseStatement(eventTypeConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.EXPRESSION: {
				Expression expression = (Expression)theEObject;
				T result = caseExpression(expression);
				if (result == null) result = caseStatement(expression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.FUNCTION: {
				Function function = (Function)theEObject;
				T result = caseFunction(function);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.FUNCTION_CALL: {
				FunctionCall functionCall = (FunctionCall)theEObject;
				T result = caseFunctionCall(functionCall);
				if (result == null) result = caseCallStatement(functionCall);
				if (result == null) result = caseStatement(functionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.FOR_STATEMENT: {
				ForStatement forStatement = (ForStatement)theEObject;
				T result = caseForStatement(forStatement);
				if (result == null) result = caseControlStructure(forStatement);
				if (result == null) result = caseStatement(forStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.GO_TO_STATEMENT: {
				GoToStatement goToStatement = (GoToStatement)theEObject;
				T result = caseGoToStatement(goToStatement);
				if (result == null) result = caseControlStructure(goToStatement);
				if (result == null) result = caseStatement(goToStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.IF_STATEMENT: {
				IfStatement ifStatement = (IfStatement)theEObject;
				T result = caseIfStatement(ifStatement);
				if (result == null) result = caseControlStructure(ifStatement);
				if (result == null) result = caseStatement(ifStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.INTEGER_CONSTANT: {
				IntegerConstant integerConstant = (IntegerConstant)theEObject;
				T result = caseIntegerConstant(integerConstant);
				if (result == null) result = caseConstantExpression(integerConstant);
				if (result == null) result = caseValueExpression(integerConstant);
				if (result == null) result = caseExpression(integerConstant);
				if (result == null) result = caseStatement(integerConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.LABEL: {
				Label label = (Label)theEObject;
				T result = caseLabel(label);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.MONITOR_HANDLER: {
				MonitorHandler monitorHandler = (MonitorHandler)theEObject;
				T result = caseMonitorHandler(monitorHandler);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.MONITOR_STATEMENT: {
				MonitorStatement monitorStatement = (MonitorStatement)theEObject;
				T result = caseMonitorStatement(monitorStatement);
				if (result == null) result = caseStatement(monitorStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.NULLARY_BUILT_IN_FUNCTION_CALL: {
				NullaryBuiltInFunctionCall nullaryBuiltInFunctionCall = (NullaryBuiltInFunctionCall)theEObject;
				T result = caseNullaryBuiltInFunctionCall(nullaryBuiltInFunctionCall);
				if (result == null) result = caseBuiltInFunctionCall(nullaryBuiltInFunctionCall);
				if (result == null) result = caseCallStatement(nullaryBuiltInFunctionCall);
				if (result == null) result = caseStatement(nullaryBuiltInFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.NULLARY_BUILT_IN_VALUE_FUNCTION_CALL: {
				NullaryBuiltInValueFunctionCall nullaryBuiltInValueFunctionCall = (NullaryBuiltInValueFunctionCall)theEObject;
				T result = caseNullaryBuiltInValueFunctionCall(nullaryBuiltInValueFunctionCall);
				if (result == null) result = caseBuiltInValueFunctionCall(nullaryBuiltInValueFunctionCall);
				if (result == null) result = caseExpression(nullaryBuiltInValueFunctionCall);
				if (result == null) result = caseStatement(nullaryBuiltInValueFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.OUTPUT_MODE_CONSTANT: {
				OutputModeConstant outputModeConstant = (OutputModeConstant)theEObject;
				T result = caseOutputModeConstant(outputModeConstant);
				if (result == null) result = caseConstantExpression(outputModeConstant);
				if (result == null) result = caseValueExpression(outputModeConstant);
				if (result == null) result = caseExpression(outputModeConstant);
				if (result == null) result = caseStatement(outputModeConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.OUTPUT_PORT_NAME_CONSTANT: {
				OutputPortNameConstant outputPortNameConstant = (OutputPortNameConstant)theEObject;
				T result = caseOutputPortNameConstant(outputPortNameConstant);
				if (result == null) result = caseConstantExpression(outputPortNameConstant);
				if (result == null) result = caseValueExpression(outputPortNameConstant);
				if (result == null) result = caseExpression(outputPortNameConstant);
				if (result == null) result = caseStatement(outputPortNameConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.PARAMETER: {
				Parameter parameter = (Parameter)theEObject;
				T result = caseParameter(parameter);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.PROGRAM: {
				Program program = (Program)theEObject;
				T result = caseProgram(program);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.PROGRAMS: {
				Programs programs = (Programs)theEObject;
				T result = casePrograms(programs);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL: {
				QuaternaryBuiltInFunctionCall quaternaryBuiltInFunctionCall = (QuaternaryBuiltInFunctionCall)theEObject;
				T result = caseQuaternaryBuiltInFunctionCall(quaternaryBuiltInFunctionCall);
				if (result == null) result = caseBuiltInFunctionCall(quaternaryBuiltInFunctionCall);
				if (result == null) result = caseCallStatement(quaternaryBuiltInFunctionCall);
				if (result == null) result = caseStatement(quaternaryBuiltInFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.REPEAT_STATEMENT: {
				RepeatStatement repeatStatement = (RepeatStatement)theEObject;
				T result = caseRepeatStatement(repeatStatement);
				if (result == null) result = caseControlStructure(repeatStatement);
				if (result == null) result = caseStatement(repeatStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.RETURN_STATEMENT: {
				ReturnStatement returnStatement = (ReturnStatement)theEObject;
				T result = caseReturnStatement(returnStatement);
				if (result == null) result = caseStatement(returnStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SENARY_BUILT_IN_FUNCTION_CALL: {
				SenaryBuiltInFunctionCall senaryBuiltInFunctionCall = (SenaryBuiltInFunctionCall)theEObject;
				T result = caseSenaryBuiltInFunctionCall(senaryBuiltInFunctionCall);
				if (result == null) result = caseBuiltInFunctionCall(senaryBuiltInFunctionCall);
				if (result == null) result = caseCallStatement(senaryBuiltInFunctionCall);
				if (result == null) result = caseStatement(senaryBuiltInFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SENSOR_CONFIG_CONSTANT: {
				SensorConfigConstant sensorConfigConstant = (SensorConfigConstant)theEObject;
				T result = caseSensorConfigConstant(sensorConfigConstant);
				if (result == null) result = caseConstantExpression(sensorConfigConstant);
				if (result == null) result = caseValueExpression(sensorConfigConstant);
				if (result == null) result = caseExpression(sensorConfigConstant);
				if (result == null) result = caseStatement(sensorConfigConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SENSOR_MODE_CONSTANT: {
				SensorModeConstant sensorModeConstant = (SensorModeConstant)theEObject;
				T result = caseSensorModeConstant(sensorModeConstant);
				if (result == null) result = caseConstantExpression(sensorModeConstant);
				if (result == null) result = caseValueExpression(sensorModeConstant);
				if (result == null) result = caseExpression(sensorModeConstant);
				if (result == null) result = caseStatement(sensorModeConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SENSOR_NAME_CONSTANT: {
				SensorNameConstant sensorNameConstant = (SensorNameConstant)theEObject;
				T result = caseSensorNameConstant(sensorNameConstant);
				if (result == null) result = caseConstantExpression(sensorNameConstant);
				if (result == null) result = caseValueExpression(sensorNameConstant);
				if (result == null) result = caseExpression(sensorNameConstant);
				if (result == null) result = caseStatement(sensorNameConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SENSOR_TYPE_CONSTANT: {
				SensorTypeConstant sensorTypeConstant = (SensorTypeConstant)theEObject;
				T result = caseSensorTypeConstant(sensorTypeConstant);
				if (result == null) result = caseConstantExpression(sensorTypeConstant);
				if (result == null) result = caseValueExpression(sensorTypeConstant);
				if (result == null) result = caseExpression(sensorTypeConstant);
				if (result == null) result = caseStatement(sensorTypeConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SERIAL_BAUD_CONSTANT: {
				SerialBaudConstant serialBaudConstant = (SerialBaudConstant)theEObject;
				T result = caseSerialBaudConstant(serialBaudConstant);
				if (result == null) result = caseConstantExpression(serialBaudConstant);
				if (result == null) result = caseValueExpression(serialBaudConstant);
				if (result == null) result = caseExpression(serialBaudConstant);
				if (result == null) result = caseStatement(serialBaudConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SERIAL_BIPHASE_CONSTANT: {
				SerialBiphaseConstant serialBiphaseConstant = (SerialBiphaseConstant)theEObject;
				T result = caseSerialBiphaseConstant(serialBiphaseConstant);
				if (result == null) result = caseConstantExpression(serialBiphaseConstant);
				if (result == null) result = caseValueExpression(serialBiphaseConstant);
				if (result == null) result = caseExpression(serialBiphaseConstant);
				if (result == null) result = caseStatement(serialBiphaseConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SERIAL_CHECKSUM_CONSTANT: {
				SerialChecksumConstant serialChecksumConstant = (SerialChecksumConstant)theEObject;
				T result = caseSerialChecksumConstant(serialChecksumConstant);
				if (result == null) result = caseConstantExpression(serialChecksumConstant);
				if (result == null) result = caseValueExpression(serialChecksumConstant);
				if (result == null) result = caseExpression(serialChecksumConstant);
				if (result == null) result = caseStatement(serialChecksumConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SERIAL_CHANNEL_CONSTANT: {
				SerialChannelConstant serialChannelConstant = (SerialChannelConstant)theEObject;
				T result = caseSerialChannelConstant(serialChannelConstant);
				if (result == null) result = caseConstantExpression(serialChannelConstant);
				if (result == null) result = caseValueExpression(serialChannelConstant);
				if (result == null) result = caseExpression(serialChannelConstant);
				if (result == null) result = caseStatement(serialChannelConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SERIAL_COMM_CONSTANT: {
				SerialCommConstant serialCommConstant = (SerialCommConstant)theEObject;
				T result = caseSerialCommConstant(serialCommConstant);
				if (result == null) result = caseConstantExpression(serialCommConstant);
				if (result == null) result = caseValueExpression(serialCommConstant);
				if (result == null) result = caseExpression(serialCommConstant);
				if (result == null) result = caseStatement(serialCommConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SERIAL_PACKET_CONSTANT: {
				SerialPacketConstant serialPacketConstant = (SerialPacketConstant)theEObject;
				T result = caseSerialPacketConstant(serialPacketConstant);
				if (result == null) result = caseConstantExpression(serialPacketConstant);
				if (result == null) result = caseValueExpression(serialPacketConstant);
				if (result == null) result = caseExpression(serialPacketConstant);
				if (result == null) result = caseStatement(serialPacketConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SOUND_CONSTANT: {
				SoundConstant soundConstant = (SoundConstant)theEObject;
				T result = caseSoundConstant(soundConstant);
				if (result == null) result = caseConstantExpression(soundConstant);
				if (result == null) result = caseValueExpression(soundConstant);
				if (result == null) result = caseExpression(soundConstant);
				if (result == null) result = caseStatement(soundConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.STATEMENT: {
				Statement statement = (Statement)theEObject;
				T result = caseStatement(statement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.START_STATEMENT: {
				StartStatement startStatement = (StartStatement)theEObject;
				T result = caseStartStatement(startStatement);
				if (result == null) result = caseStatement(startStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.STOP_STATEMENT: {
				StopStatement stopStatement = (StopStatement)theEObject;
				T result = caseStopStatement(stopStatement);
				if (result == null) result = caseStatement(stopStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SUBROUTINE: {
				Subroutine subroutine = (Subroutine)theEObject;
				T result = caseSubroutine(subroutine);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SUBROUTINE_CALL: {
				SubroutineCall subroutineCall = (SubroutineCall)theEObject;
				T result = caseSubroutineCall(subroutineCall);
				if (result == null) result = caseCallStatement(subroutineCall);
				if (result == null) result = caseStatement(subroutineCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.SWITCH_STATEMENT: {
				SwitchStatement switchStatement = (SwitchStatement)theEObject;
				T result = caseSwitchStatement(switchStatement);
				if (result == null) result = caseControlStructure(switchStatement);
				if (result == null) result = caseStatement(switchStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.TASK: {
				Task task = (Task)theEObject;
				T result = caseTask(task);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL: {
				TernaryBuiltInFunctionCall ternaryBuiltInFunctionCall = (TernaryBuiltInFunctionCall)theEObject;
				T result = caseTernaryBuiltInFunctionCall(ternaryBuiltInFunctionCall);
				if (result == null) result = caseBuiltInFunctionCall(ternaryBuiltInFunctionCall);
				if (result == null) result = caseCallStatement(ternaryBuiltInFunctionCall);
				if (result == null) result = caseStatement(ternaryBuiltInFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.TERNARY_EXPRESSION: {
				TernaryExpression ternaryExpression = (TernaryExpression)theEObject;
				T result = caseTernaryExpression(ternaryExpression);
				if (result == null) result = caseCompoundExpression(ternaryExpression);
				if (result == null) result = caseExpression(ternaryExpression);
				if (result == null) result = caseStatement(ternaryExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.TX_POWER_CONSTANT: {
				TxPowerConstant txPowerConstant = (TxPowerConstant)theEObject;
				T result = caseTxPowerConstant(txPowerConstant);
				if (result == null) result = caseConstantExpression(txPowerConstant);
				if (result == null) result = caseValueExpression(txPowerConstant);
				if (result == null) result = caseExpression(txPowerConstant);
				if (result == null) result = caseStatement(txPowerConstant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL: {
				UnaryBuiltInFunctionCall unaryBuiltInFunctionCall = (UnaryBuiltInFunctionCall)theEObject;
				T result = caseUnaryBuiltInFunctionCall(unaryBuiltInFunctionCall);
				if (result == null) result = caseBuiltInFunctionCall(unaryBuiltInFunctionCall);
				if (result == null) result = caseCallStatement(unaryBuiltInFunctionCall);
				if (result == null) result = caseStatement(unaryBuiltInFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.UNARY_BUILT_IN_VALUE_FUNCTION_CALL: {
				UnaryBuiltInValueFunctionCall unaryBuiltInValueFunctionCall = (UnaryBuiltInValueFunctionCall)theEObject;
				T result = caseUnaryBuiltInValueFunctionCall(unaryBuiltInValueFunctionCall);
				if (result == null) result = caseBuiltInValueFunctionCall(unaryBuiltInValueFunctionCall);
				if (result == null) result = caseExpression(unaryBuiltInValueFunctionCall);
				if (result == null) result = caseStatement(unaryBuiltInValueFunctionCall);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.UNARY_EXPRESSION: {
				UnaryExpression unaryExpression = (UnaryExpression)theEObject;
				T result = caseUnaryExpression(unaryExpression);
				if (result == null) result = caseCompoundExpression(unaryExpression);
				if (result == null) result = caseExpression(unaryExpression);
				if (result == null) result = caseStatement(unaryExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.UNTIL_STATEMENT: {
				UntilStatement untilStatement = (UntilStatement)theEObject;
				T result = caseUntilStatement(untilStatement);
				if (result == null) result = caseControlStructure(untilStatement);
				if (result == null) result = caseStatement(untilStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.VALUE_EXPRESSION: {
				ValueExpression valueExpression = (ValueExpression)theEObject;
				T result = caseValueExpression(valueExpression);
				if (result == null) result = caseExpression(valueExpression);
				if (result == null) result = caseStatement(valueExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.VARIABLE: {
				Variable variable = (Variable)theEObject;
				T result = caseVariable(variable);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.VARIABLE_EXPRESSION: {
				VariableExpression variableExpression = (VariableExpression)theEObject;
				T result = caseVariableExpression(variableExpression);
				if (result == null) result = caseValueExpression(variableExpression);
				if (result == null) result = caseExpression(variableExpression);
				if (result == null) result = caseStatement(variableExpression);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case NqcPackage.WHILE_STATEMENT: {
				WhileStatement whileStatement = (WhileStatement)theEObject;
				T result = caseWhileStatement(whileStatement);
				if (result == null) result = caseControlStructure(whileStatement);
				if (result == null) result = caseStatement(whileStatement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			default: return defaultCase(theEObject);
		}
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Acquire Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Acquire Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseAcquireConstant(AcquireConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Acquire Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Acquire Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseAcquireStatement(AcquireStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Array Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Array Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseArrayExpression(ArrayExpression object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Assignment Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Assignment Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseAssignmentStatement(AssignmentStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Binary Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Binary Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseBinaryExpression(BinaryExpression object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Binary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Binary Built In Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseBinaryBuiltInFunctionCall(BinaryBuiltInFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Binary Built In Value Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Binary Built In Value Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseBinaryBuiltInValueFunctionCall(BinaryBuiltInValueFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Block Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Block Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseBlockStatement(BlockStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Boolean Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Boolean Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseBooleanConstant(BooleanConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Break Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Break Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseBreakStatement(BreakStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Built In Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseBuiltInFunctionCall(BuiltInFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Built In Value Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Built In Value Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseBuiltInValueFunctionCall(BuiltInValueFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Call Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Call Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCallStatement(CallStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Case</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Case</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCase(Case object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Compound Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Compound Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCompoundExpression(CompoundExpression object) {
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
	 * Returns the result of interpreting the object as an instance of '<em>Continue Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Continue Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseContinueStatement(ContinueStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Control Structure</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Control Structure</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseControlStructure(ControlStructure object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Direction Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Direction Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseDirectionConstant(DirectionConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Display Mode Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Display Mode Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseDisplayModeConstant(DisplayModeConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Do While Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Do While Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseDoWhileStatement(DoWhileStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Empty Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Empty Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseEmptyStatement(EmptyStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Event Type Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Event Type Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseEventTypeConstant(EventTypeConstant object) {
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
	 * Returns the result of interpreting the object as an instance of '<em>Function</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Function</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseFunction(Function object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseFunctionCall(FunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>For Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>For Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseForStatement(ForStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Go To Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Go To Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseGoToStatement(GoToStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>If Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>If Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseIfStatement(IfStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Integer Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Integer Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseIntegerConstant(IntegerConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Label</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Label</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseLabel(Label object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Monitor Handler</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Monitor Handler</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseMonitorHandler(MonitorHandler object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Monitor Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Monitor Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseMonitorStatement(MonitorStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Nullary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Nullary Built In Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseNullaryBuiltInFunctionCall(NullaryBuiltInFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Nullary Built In Value Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Nullary Built In Value Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseNullaryBuiltInValueFunctionCall(NullaryBuiltInValueFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Output Mode Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Output Mode Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseOutputModeConstant(OutputModeConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Output Port Name Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Output Port Name Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseOutputPortNameConstant(OutputPortNameConstant object) {
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
	 * Returns the result of interpreting the object as an instance of '<em>Program</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Program</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseProgram(Program object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Programs</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Programs</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casePrograms(Programs object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Quaternary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Quaternary Built In Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseQuaternaryBuiltInFunctionCall(QuaternaryBuiltInFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Repeat Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Repeat Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseRepeatStatement(RepeatStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Return Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Return Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseReturnStatement(ReturnStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Senary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Senary Built In Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSenaryBuiltInFunctionCall(SenaryBuiltInFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Sensor Config Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Sensor Config Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSensorConfigConstant(SensorConfigConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Sensor Mode Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Sensor Mode Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSensorModeConstant(SensorModeConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Sensor Name Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Sensor Name Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSensorNameConstant(SensorNameConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Sensor Type Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Sensor Type Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSensorTypeConstant(SensorTypeConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Serial Baud Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Serial Baud Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSerialBaudConstant(SerialBaudConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Serial Biphase Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Serial Biphase Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSerialBiphaseConstant(SerialBiphaseConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Serial Checksum Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Serial Checksum Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSerialChecksumConstant(SerialChecksumConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Serial Channel Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Serial Channel Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSerialChannelConstant(SerialChannelConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Serial Comm Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Serial Comm Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSerialCommConstant(SerialCommConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Serial Packet Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Serial Packet Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSerialPacketConstant(SerialPacketConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Sound Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Sound Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSoundConstant(SoundConstant object) {
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
	 * Returns the result of interpreting the object as an instance of '<em>Start Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Start Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseStartStatement(StartStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Stop Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Stop Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseStopStatement(StopStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Subroutine</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Subroutine</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSubroutine(Subroutine object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Subroutine Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Subroutine Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSubroutineCall(SubroutineCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Switch Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Switch Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSwitchStatement(SwitchStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Task</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Task</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseTask(Task object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Ternary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Ternary Built In Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseTernaryBuiltInFunctionCall(TernaryBuiltInFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Ternary Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Ternary Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseTernaryExpression(TernaryExpression object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Tx Power Constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Tx Power Constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseTxPowerConstant(TxPowerConstant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Unary Built In Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Unary Built In Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseUnaryBuiltInFunctionCall(UnaryBuiltInFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Unary Built In Value Function Call</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Unary Built In Value Function Call</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseUnaryBuiltInValueFunctionCall(UnaryBuiltInValueFunctionCall object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Unary Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Unary Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseUnaryExpression(UnaryExpression object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Until Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Until Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseUntilStatement(UntilStatement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Value Expression</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Value Expression</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseValueExpression(ValueExpression object) {
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
	 * Returns the result of interpreting the object as an instance of '<em>While Statement</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>While Statement</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseWhileStatement(WhileStatement object) {
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

} //NqcSwitch
