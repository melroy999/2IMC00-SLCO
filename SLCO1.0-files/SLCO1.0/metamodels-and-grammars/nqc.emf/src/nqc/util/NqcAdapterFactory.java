/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.util;

import nqc.*;

import org.eclipse.emf.common.notify.Adapter;
import org.eclipse.emf.common.notify.Notifier;

import org.eclipse.emf.common.notify.impl.AdapterFactoryImpl;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * The <b>Adapter Factory</b> for the model.
 * It provides an adapter <code>createXXX</code> method for each class of the model.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage
 * @generated
 */
public class NqcAdapterFactory extends AdapterFactoryImpl {
	/**
	 * The cached model package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static NqcPackage modelPackage;

	/**
	 * Creates an instance of the adapter factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NqcAdapterFactory() {
		if (modelPackage == null) {
			modelPackage = NqcPackage.eINSTANCE;
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
	protected NqcSwitch<Adapter> modelSwitch =
		new NqcSwitch<Adapter>() {
			@Override
			public Adapter caseAcquireConstant(AcquireConstant object) {
				return createAcquireConstantAdapter();
			}
			@Override
			public Adapter caseAcquireStatement(AcquireStatement object) {
				return createAcquireStatementAdapter();
			}
			@Override
			public Adapter caseArrayExpression(ArrayExpression object) {
				return createArrayExpressionAdapter();
			}
			@Override
			public Adapter caseAssignmentStatement(AssignmentStatement object) {
				return createAssignmentStatementAdapter();
			}
			@Override
			public Adapter caseBinaryExpression(BinaryExpression object) {
				return createBinaryExpressionAdapter();
			}
			@Override
			public Adapter caseBinaryBuiltInFunctionCall(BinaryBuiltInFunctionCall object) {
				return createBinaryBuiltInFunctionCallAdapter();
			}
			@Override
			public Adapter caseBinaryBuiltInValueFunctionCall(BinaryBuiltInValueFunctionCall object) {
				return createBinaryBuiltInValueFunctionCallAdapter();
			}
			@Override
			public Adapter caseBlockStatement(BlockStatement object) {
				return createBlockStatementAdapter();
			}
			@Override
			public Adapter caseBooleanConstant(BooleanConstant object) {
				return createBooleanConstantAdapter();
			}
			@Override
			public Adapter caseBreakStatement(BreakStatement object) {
				return createBreakStatementAdapter();
			}
			@Override
			public Adapter caseBuiltInFunctionCall(BuiltInFunctionCall object) {
				return createBuiltInFunctionCallAdapter();
			}
			@Override
			public Adapter caseBuiltInValueFunctionCall(BuiltInValueFunctionCall object) {
				return createBuiltInValueFunctionCallAdapter();
			}
			@Override
			public Adapter caseCallStatement(CallStatement object) {
				return createCallStatementAdapter();
			}
			@Override
			public Adapter caseCase(Case object) {
				return createCaseAdapter();
			}
			@Override
			public Adapter caseCompoundExpression(CompoundExpression object) {
				return createCompoundExpressionAdapter();
			}
			@Override
			public Adapter caseConstantExpression(ConstantExpression object) {
				return createConstantExpressionAdapter();
			}
			@Override
			public Adapter caseContinueStatement(ContinueStatement object) {
				return createContinueStatementAdapter();
			}
			@Override
			public Adapter caseControlStructure(ControlStructure object) {
				return createControlStructureAdapter();
			}
			@Override
			public Adapter caseDirectionConstant(DirectionConstant object) {
				return createDirectionConstantAdapter();
			}
			@Override
			public Adapter caseDisplayModeConstant(DisplayModeConstant object) {
				return createDisplayModeConstantAdapter();
			}
			@Override
			public Adapter caseDoWhileStatement(DoWhileStatement object) {
				return createDoWhileStatementAdapter();
			}
			@Override
			public Adapter caseEmptyStatement(EmptyStatement object) {
				return createEmptyStatementAdapter();
			}
			@Override
			public Adapter caseEventTypeConstant(EventTypeConstant object) {
				return createEventTypeConstantAdapter();
			}
			@Override
			public Adapter caseExpression(Expression object) {
				return createExpressionAdapter();
			}
			@Override
			public Adapter caseFunction(Function object) {
				return createFunctionAdapter();
			}
			@Override
			public Adapter caseFunctionCall(FunctionCall object) {
				return createFunctionCallAdapter();
			}
			@Override
			public Adapter caseForStatement(ForStatement object) {
				return createForStatementAdapter();
			}
			@Override
			public Adapter caseGoToStatement(GoToStatement object) {
				return createGoToStatementAdapter();
			}
			@Override
			public Adapter caseIfStatement(IfStatement object) {
				return createIfStatementAdapter();
			}
			@Override
			public Adapter caseIntegerConstant(IntegerConstant object) {
				return createIntegerConstantAdapter();
			}
			@Override
			public Adapter caseLabel(Label object) {
				return createLabelAdapter();
			}
			@Override
			public Adapter caseMonitorHandler(MonitorHandler object) {
				return createMonitorHandlerAdapter();
			}
			@Override
			public Adapter caseMonitorStatement(MonitorStatement object) {
				return createMonitorStatementAdapter();
			}
			@Override
			public Adapter caseNullaryBuiltInFunctionCall(NullaryBuiltInFunctionCall object) {
				return createNullaryBuiltInFunctionCallAdapter();
			}
			@Override
			public Adapter caseNullaryBuiltInValueFunctionCall(NullaryBuiltInValueFunctionCall object) {
				return createNullaryBuiltInValueFunctionCallAdapter();
			}
			@Override
			public Adapter caseOutputModeConstant(OutputModeConstant object) {
				return createOutputModeConstantAdapter();
			}
			@Override
			public Adapter caseOutputPortNameConstant(OutputPortNameConstant object) {
				return createOutputPortNameConstantAdapter();
			}
			@Override
			public Adapter caseParameter(Parameter object) {
				return createParameterAdapter();
			}
			@Override
			public Adapter caseProgram(Program object) {
				return createProgramAdapter();
			}
			@Override
			public Adapter casePrograms(Programs object) {
				return createProgramsAdapter();
			}
			@Override
			public Adapter caseQuaternaryBuiltInFunctionCall(QuaternaryBuiltInFunctionCall object) {
				return createQuaternaryBuiltInFunctionCallAdapter();
			}
			@Override
			public Adapter caseRepeatStatement(RepeatStatement object) {
				return createRepeatStatementAdapter();
			}
			@Override
			public Adapter caseReturnStatement(ReturnStatement object) {
				return createReturnStatementAdapter();
			}
			@Override
			public Adapter caseSenaryBuiltInFunctionCall(SenaryBuiltInFunctionCall object) {
				return createSenaryBuiltInFunctionCallAdapter();
			}
			@Override
			public Adapter caseSensorConfigConstant(SensorConfigConstant object) {
				return createSensorConfigConstantAdapter();
			}
			@Override
			public Adapter caseSensorModeConstant(SensorModeConstant object) {
				return createSensorModeConstantAdapter();
			}
			@Override
			public Adapter caseSensorNameConstant(SensorNameConstant object) {
				return createSensorNameConstantAdapter();
			}
			@Override
			public Adapter caseSensorTypeConstant(SensorTypeConstant object) {
				return createSensorTypeConstantAdapter();
			}
			@Override
			public Adapter caseSerialBaudConstant(SerialBaudConstant object) {
				return createSerialBaudConstantAdapter();
			}
			@Override
			public Adapter caseSerialBiphaseConstant(SerialBiphaseConstant object) {
				return createSerialBiphaseConstantAdapter();
			}
			@Override
			public Adapter caseSerialChecksumConstant(SerialChecksumConstant object) {
				return createSerialChecksumConstantAdapter();
			}
			@Override
			public Adapter caseSerialChannelConstant(SerialChannelConstant object) {
				return createSerialChannelConstantAdapter();
			}
			@Override
			public Adapter caseSerialCommConstant(SerialCommConstant object) {
				return createSerialCommConstantAdapter();
			}
			@Override
			public Adapter caseSerialPacketConstant(SerialPacketConstant object) {
				return createSerialPacketConstantAdapter();
			}
			@Override
			public Adapter caseSoundConstant(SoundConstant object) {
				return createSoundConstantAdapter();
			}
			@Override
			public Adapter caseStatement(Statement object) {
				return createStatementAdapter();
			}
			@Override
			public Adapter caseStartStatement(StartStatement object) {
				return createStartStatementAdapter();
			}
			@Override
			public Adapter caseStopStatement(StopStatement object) {
				return createStopStatementAdapter();
			}
			@Override
			public Adapter caseSubroutine(Subroutine object) {
				return createSubroutineAdapter();
			}
			@Override
			public Adapter caseSubroutineCall(SubroutineCall object) {
				return createSubroutineCallAdapter();
			}
			@Override
			public Adapter caseSwitchStatement(SwitchStatement object) {
				return createSwitchStatementAdapter();
			}
			@Override
			public Adapter caseTask(Task object) {
				return createTaskAdapter();
			}
			@Override
			public Adapter caseTernaryBuiltInFunctionCall(TernaryBuiltInFunctionCall object) {
				return createTernaryBuiltInFunctionCallAdapter();
			}
			@Override
			public Adapter caseTernaryExpression(TernaryExpression object) {
				return createTernaryExpressionAdapter();
			}
			@Override
			public Adapter caseTxPowerConstant(TxPowerConstant object) {
				return createTxPowerConstantAdapter();
			}
			@Override
			public Adapter caseUnaryBuiltInFunctionCall(UnaryBuiltInFunctionCall object) {
				return createUnaryBuiltInFunctionCallAdapter();
			}
			@Override
			public Adapter caseUnaryBuiltInValueFunctionCall(UnaryBuiltInValueFunctionCall object) {
				return createUnaryBuiltInValueFunctionCallAdapter();
			}
			@Override
			public Adapter caseUnaryExpression(UnaryExpression object) {
				return createUnaryExpressionAdapter();
			}
			@Override
			public Adapter caseUntilStatement(UntilStatement object) {
				return createUntilStatementAdapter();
			}
			@Override
			public Adapter caseValueExpression(ValueExpression object) {
				return createValueExpressionAdapter();
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
			public Adapter caseWhileStatement(WhileStatement object) {
				return createWhileStatementAdapter();
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
	 * Creates a new adapter for an object of class '{@link nqc.AcquireConstant <em>Acquire Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.AcquireConstant
	 * @generated
	 */
	public Adapter createAcquireConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.AcquireStatement <em>Acquire Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.AcquireStatement
	 * @generated
	 */
	public Adapter createAcquireStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.ArrayExpression <em>Array Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.ArrayExpression
	 * @generated
	 */
	public Adapter createArrayExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.AssignmentStatement <em>Assignment Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.AssignmentStatement
	 * @generated
	 */
	public Adapter createAssignmentStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.BinaryExpression <em>Binary Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.BinaryExpression
	 * @generated
	 */
	public Adapter createBinaryExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.BinaryBuiltInFunctionCall <em>Binary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.BinaryBuiltInFunctionCall
	 * @generated
	 */
	public Adapter createBinaryBuiltInFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.BinaryBuiltInValueFunctionCall <em>Binary Built In Value Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.BinaryBuiltInValueFunctionCall
	 * @generated
	 */
	public Adapter createBinaryBuiltInValueFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.BlockStatement <em>Block Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.BlockStatement
	 * @generated
	 */
	public Adapter createBlockStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.BooleanConstant <em>Boolean Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.BooleanConstant
	 * @generated
	 */
	public Adapter createBooleanConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.BreakStatement <em>Break Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.BreakStatement
	 * @generated
	 */
	public Adapter createBreakStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.BuiltInFunctionCall <em>Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.BuiltInFunctionCall
	 * @generated
	 */
	public Adapter createBuiltInFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.BuiltInValueFunctionCall <em>Built In Value Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.BuiltInValueFunctionCall
	 * @generated
	 */
	public Adapter createBuiltInValueFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.CallStatement <em>Call Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.CallStatement
	 * @generated
	 */
	public Adapter createCallStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Case <em>Case</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Case
	 * @generated
	 */
	public Adapter createCaseAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.CompoundExpression <em>Compound Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.CompoundExpression
	 * @generated
	 */
	public Adapter createCompoundExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.ConstantExpression <em>Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.ConstantExpression
	 * @generated
	 */
	public Adapter createConstantExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.ContinueStatement <em>Continue Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.ContinueStatement
	 * @generated
	 */
	public Adapter createContinueStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.ControlStructure <em>Control Structure</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.ControlStructure
	 * @generated
	 */
	public Adapter createControlStructureAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.DirectionConstant <em>Direction Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.DirectionConstant
	 * @generated
	 */
	public Adapter createDirectionConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.DisplayModeConstant <em>Display Mode Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.DisplayModeConstant
	 * @generated
	 */
	public Adapter createDisplayModeConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.DoWhileStatement <em>Do While Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.DoWhileStatement
	 * @generated
	 */
	public Adapter createDoWhileStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.EmptyStatement <em>Empty Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.EmptyStatement
	 * @generated
	 */
	public Adapter createEmptyStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.EventTypeConstant <em>Event Type Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.EventTypeConstant
	 * @generated
	 */
	public Adapter createEventTypeConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Expression <em>Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Expression
	 * @generated
	 */
	public Adapter createExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Function <em>Function</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Function
	 * @generated
	 */
	public Adapter createFunctionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.FunctionCall <em>Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.FunctionCall
	 * @generated
	 */
	public Adapter createFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.ForStatement <em>For Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.ForStatement
	 * @generated
	 */
	public Adapter createForStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.GoToStatement <em>Go To Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.GoToStatement
	 * @generated
	 */
	public Adapter createGoToStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.IfStatement <em>If Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.IfStatement
	 * @generated
	 */
	public Adapter createIfStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.IntegerConstant <em>Integer Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.IntegerConstant
	 * @generated
	 */
	public Adapter createIntegerConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Label <em>Label</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Label
	 * @generated
	 */
	public Adapter createLabelAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.MonitorHandler <em>Monitor Handler</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.MonitorHandler
	 * @generated
	 */
	public Adapter createMonitorHandlerAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.MonitorStatement <em>Monitor Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.MonitorStatement
	 * @generated
	 */
	public Adapter createMonitorStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.NullaryBuiltInFunctionCall <em>Nullary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.NullaryBuiltInFunctionCall
	 * @generated
	 */
	public Adapter createNullaryBuiltInFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.NullaryBuiltInValueFunctionCall <em>Nullary Built In Value Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.NullaryBuiltInValueFunctionCall
	 * @generated
	 */
	public Adapter createNullaryBuiltInValueFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.OutputModeConstant <em>Output Mode Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.OutputModeConstant
	 * @generated
	 */
	public Adapter createOutputModeConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.OutputPortNameConstant <em>Output Port Name Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.OutputPortNameConstant
	 * @generated
	 */
	public Adapter createOutputPortNameConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Parameter <em>Parameter</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Parameter
	 * @generated
	 */
	public Adapter createParameterAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Program <em>Program</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Program
	 * @generated
	 */
	public Adapter createProgramAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Programs <em>Programs</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Programs
	 * @generated
	 */
	public Adapter createProgramsAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.QuaternaryBuiltInFunctionCall <em>Quaternary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.QuaternaryBuiltInFunctionCall
	 * @generated
	 */
	public Adapter createQuaternaryBuiltInFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.RepeatStatement <em>Repeat Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.RepeatStatement
	 * @generated
	 */
	public Adapter createRepeatStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.ReturnStatement <em>Return Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.ReturnStatement
	 * @generated
	 */
	public Adapter createReturnStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SenaryBuiltInFunctionCall <em>Senary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SenaryBuiltInFunctionCall
	 * @generated
	 */
	public Adapter createSenaryBuiltInFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SensorConfigConstant <em>Sensor Config Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SensorConfigConstant
	 * @generated
	 */
	public Adapter createSensorConfigConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SensorModeConstant <em>Sensor Mode Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SensorModeConstant
	 * @generated
	 */
	public Adapter createSensorModeConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SensorNameConstant <em>Sensor Name Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SensorNameConstant
	 * @generated
	 */
	public Adapter createSensorNameConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SensorTypeConstant <em>Sensor Type Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SensorTypeConstant
	 * @generated
	 */
	public Adapter createSensorTypeConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SerialBaudConstant <em>Serial Baud Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SerialBaudConstant
	 * @generated
	 */
	public Adapter createSerialBaudConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SerialBiphaseConstant <em>Serial Biphase Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SerialBiphaseConstant
	 * @generated
	 */
	public Adapter createSerialBiphaseConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SerialChecksumConstant <em>Serial Checksum Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SerialChecksumConstant
	 * @generated
	 */
	public Adapter createSerialChecksumConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SerialChannelConstant <em>Serial Channel Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SerialChannelConstant
	 * @generated
	 */
	public Adapter createSerialChannelConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SerialCommConstant <em>Serial Comm Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SerialCommConstant
	 * @generated
	 */
	public Adapter createSerialCommConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SerialPacketConstant <em>Serial Packet Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SerialPacketConstant
	 * @generated
	 */
	public Adapter createSerialPacketConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SoundConstant <em>Sound Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SoundConstant
	 * @generated
	 */
	public Adapter createSoundConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Statement <em>Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Statement
	 * @generated
	 */
	public Adapter createStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.StartStatement <em>Start Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.StartStatement
	 * @generated
	 */
	public Adapter createStartStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.StopStatement <em>Stop Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.StopStatement
	 * @generated
	 */
	public Adapter createStopStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Subroutine <em>Subroutine</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Subroutine
	 * @generated
	 */
	public Adapter createSubroutineAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SubroutineCall <em>Subroutine Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SubroutineCall
	 * @generated
	 */
	public Adapter createSubroutineCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.SwitchStatement <em>Switch Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.SwitchStatement
	 * @generated
	 */
	public Adapter createSwitchStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Task <em>Task</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Task
	 * @generated
	 */
	public Adapter createTaskAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.TernaryBuiltInFunctionCall <em>Ternary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.TernaryBuiltInFunctionCall
	 * @generated
	 */
	public Adapter createTernaryBuiltInFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.TernaryExpression <em>Ternary Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.TernaryExpression
	 * @generated
	 */
	public Adapter createTernaryExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.TxPowerConstant <em>Tx Power Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.TxPowerConstant
	 * @generated
	 */
	public Adapter createTxPowerConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.UnaryBuiltInFunctionCall <em>Unary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.UnaryBuiltInFunctionCall
	 * @generated
	 */
	public Adapter createUnaryBuiltInFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.UnaryBuiltInValueFunctionCall <em>Unary Built In Value Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.UnaryBuiltInValueFunctionCall
	 * @generated
	 */
	public Adapter createUnaryBuiltInValueFunctionCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.UnaryExpression <em>Unary Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.UnaryExpression
	 * @generated
	 */
	public Adapter createUnaryExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.UntilStatement <em>Until Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.UntilStatement
	 * @generated
	 */
	public Adapter createUntilStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.ValueExpression <em>Value Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.ValueExpression
	 * @generated
	 */
	public Adapter createValueExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.Variable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.Variable
	 * @generated
	 */
	public Adapter createVariableAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.VariableExpression <em>Variable Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.VariableExpression
	 * @generated
	 */
	public Adapter createVariableExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link nqc.WhileStatement <em>While Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see nqc.WhileStatement
	 * @generated
	 */
	public Adapter createWhileStatementAdapter() {
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

} //NqcAdapterFactory
