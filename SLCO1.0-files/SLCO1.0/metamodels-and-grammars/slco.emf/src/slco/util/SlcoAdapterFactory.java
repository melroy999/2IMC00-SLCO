/**
 */
package slco.util;

import org.eclipse.emf.common.notify.Adapter;
import org.eclipse.emf.common.notify.Notifier;

import org.eclipse.emf.common.notify.impl.AdapterFactoryImpl;

import org.eclipse.emf.ecore.EObject;

import slco.ArgumentType;
import slco.Assignment;
import slco.BidirectionalChannel;
import slco.BinaryOperatorExpression;
import slco.BooleanConstantExpression;
import slco.Channel;
import slco.ConstantExpression;
import slco.Delay;
import slco.Expression;
import slco.Final;
import slco.Initial;
import slco.IntegerConstantExpression;
import slco.Model;
import slco.Port;
import slco.SendSignal;
import slco.SignalArgument;
import slco.SignalArgumentExpression;
import slco.SignalArgumentVariable;
import slco.SignalReception;
import slco.SlcoPackage;
import slco.State;
import slco.StateMachine;
import slco.Statement;
import slco.StringConstantExpression;
import slco.TextualStatement;
import slco.Transition;
import slco.UnidirectionalChannel;
import slco.Variable;
import slco.VariableExpression;
import slco.Vertex;

/**
 * <!-- begin-user-doc -->
 * The <b>Adapter Factory</b> for the model.
 * It provides an adapter <code>createXXX</code> method for each class of the model.
 * <!-- end-user-doc -->
 * @see slco.SlcoPackage
 * @generated
 */
public class SlcoAdapterFactory extends AdapterFactoryImpl {
	/**
	 * The cached model package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static SlcoPackage modelPackage;

	/**
	 * Creates an instance of the adapter factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SlcoAdapterFactory() {
		if (modelPackage == null) {
			modelPackage = SlcoPackage.eINSTANCE;
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
	protected SlcoSwitch<Adapter> modelSwitch =
		new SlcoSwitch<Adapter>() {
			@Override
			public Adapter caseArgumentType(ArgumentType object) {
				return createArgumentTypeAdapter();
			}
			@Override
			public Adapter caseAssignment(Assignment object) {
				return createAssignmentAdapter();
			}
			@Override
			public Adapter caseBidirectionalChannel(BidirectionalChannel object) {
				return createBidirectionalChannelAdapter();
			}
			@Override
			public Adapter caseBinaryOperatorExpression(BinaryOperatorExpression object) {
				return createBinaryOperatorExpressionAdapter();
			}
			@Override
			public Adapter caseBooleanConstantExpression(BooleanConstantExpression object) {
				return createBooleanConstantExpressionAdapter();
			}
			@Override
			public Adapter caseChannel(Channel object) {
				return createChannelAdapter();
			}
			@Override
			public Adapter caseClass(slco.Class object) {
				return createClassAdapter();
			}
			@Override
			public Adapter caseConstantExpression(ConstantExpression object) {
				return createConstantExpressionAdapter();
			}
			@Override
			public Adapter caseDelay(Delay object) {
				return createDelayAdapter();
			}
			@Override
			public Adapter caseExpression(Expression object) {
				return createExpressionAdapter();
			}
			@Override
			public Adapter caseFinal(Final object) {
				return createFinalAdapter();
			}
			@Override
			public Adapter caseInitial(Initial object) {
				return createInitialAdapter();
			}
			@Override
			public Adapter caseIntegerConstantExpression(IntegerConstantExpression object) {
				return createIntegerConstantExpressionAdapter();
			}
			@Override
			public Adapter caseModel(Model object) {
				return createModelAdapter();
			}
			@Override
			public Adapter caseObject(slco.Object object) {
				return createObjectAdapter();
			}
			@Override
			public Adapter casePort(Port object) {
				return createPortAdapter();
			}
			@Override
			public Adapter caseSendSignal(SendSignal object) {
				return createSendSignalAdapter();
			}
			@Override
			public Adapter caseSignalArgument(SignalArgument object) {
				return createSignalArgumentAdapter();
			}
			@Override
			public Adapter caseSignalArgumentExpression(SignalArgumentExpression object) {
				return createSignalArgumentExpressionAdapter();
			}
			@Override
			public Adapter caseSignalArgumentVariable(SignalArgumentVariable object) {
				return createSignalArgumentVariableAdapter();
			}
			@Override
			public Adapter caseSignalReception(SignalReception object) {
				return createSignalReceptionAdapter();
			}
			@Override
			public Adapter caseState(State object) {
				return createStateAdapter();
			}
			@Override
			public Adapter caseStateMachine(StateMachine object) {
				return createStateMachineAdapter();
			}
			@Override
			public Adapter caseStatement(Statement object) {
				return createStatementAdapter();
			}
			@Override
			public Adapter caseStringConstantExpression(StringConstantExpression object) {
				return createStringConstantExpressionAdapter();
			}
			@Override
			public Adapter caseTextualStatement(TextualStatement object) {
				return createTextualStatementAdapter();
			}
			@Override
			public Adapter caseTransition(Transition object) {
				return createTransitionAdapter();
			}
			@Override
			public Adapter caseUnidirectionalChannel(UnidirectionalChannel object) {
				return createUnidirectionalChannelAdapter();
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
			public Adapter caseVertex(Vertex object) {
				return createVertexAdapter();
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
	 * Creates a new adapter for an object of class '{@link slco.ArgumentType <em>Argument Type</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.ArgumentType
	 * @generated
	 */
	public Adapter createArgumentTypeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Assignment <em>Assignment</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Assignment
	 * @generated
	 */
	public Adapter createAssignmentAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.BidirectionalChannel <em>Bidirectional Channel</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.BidirectionalChannel
	 * @generated
	 */
	public Adapter createBidirectionalChannelAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.BinaryOperatorExpression <em>Binary Operator Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.BinaryOperatorExpression
	 * @generated
	 */
	public Adapter createBinaryOperatorExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.BooleanConstantExpression <em>Boolean Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.BooleanConstantExpression
	 * @generated
	 */
	public Adapter createBooleanConstantExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Channel <em>Channel</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Channel
	 * @generated
	 */
	public Adapter createChannelAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Class <em>Class</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Class
	 * @generated
	 */
	public Adapter createClassAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.ConstantExpression <em>Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.ConstantExpression
	 * @generated
	 */
	public Adapter createConstantExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Delay <em>Delay</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Delay
	 * @generated
	 */
	public Adapter createDelayAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Expression <em>Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Expression
	 * @generated
	 */
	public Adapter createExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Final <em>Final</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Final
	 * @generated
	 */
	public Adapter createFinalAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Initial <em>Initial</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Initial
	 * @generated
	 */
	public Adapter createInitialAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.IntegerConstantExpression <em>Integer Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.IntegerConstantExpression
	 * @generated
	 */
	public Adapter createIntegerConstantExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Model <em>Model</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Model
	 * @generated
	 */
	public Adapter createModelAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Object <em>Object</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Object
	 * @generated
	 */
	public Adapter createObjectAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Port <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Port
	 * @generated
	 */
	public Adapter createPortAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.SendSignal <em>Send Signal</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.SendSignal
	 * @generated
	 */
	public Adapter createSendSignalAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.SignalArgument <em>Signal Argument</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.SignalArgument
	 * @generated
	 */
	public Adapter createSignalArgumentAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.SignalArgumentExpression <em>Signal Argument Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.SignalArgumentExpression
	 * @generated
	 */
	public Adapter createSignalArgumentExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.SignalArgumentVariable <em>Signal Argument Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.SignalArgumentVariable
	 * @generated
	 */
	public Adapter createSignalArgumentVariableAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.SignalReception <em>Signal Reception</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.SignalReception
	 * @generated
	 */
	public Adapter createSignalReceptionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.State <em>State</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.State
	 * @generated
	 */
	public Adapter createStateAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.StateMachine <em>State Machine</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.StateMachine
	 * @generated
	 */
	public Adapter createStateMachineAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Statement <em>Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Statement
	 * @generated
	 */
	public Adapter createStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.StringConstantExpression <em>String Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.StringConstantExpression
	 * @generated
	 */
	public Adapter createStringConstantExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.TextualStatement <em>Textual Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.TextualStatement
	 * @generated
	 */
	public Adapter createTextualStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Transition <em>Transition</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Transition
	 * @generated
	 */
	public Adapter createTransitionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.UnidirectionalChannel <em>Unidirectional Channel</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.UnidirectionalChannel
	 * @generated
	 */
	public Adapter createUnidirectionalChannelAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Variable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Variable
	 * @generated
	 */
	public Adapter createVariableAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.VariableExpression <em>Variable Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.VariableExpression
	 * @generated
	 */
	public Adapter createVariableExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco.Vertex <em>Vertex</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco.Vertex
	 * @generated
	 */
	public Adapter createVertexAdapter() {
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

} //SlcoAdapterFactory
