/**
 */
package slco.impl;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

import org.eclipse.emf.ecore.impl.EPackageImpl;

import slco.ArgumentType;
import slco.Assignment;
import slco.BidirectionalChannel;
import slco.BinaryOperatorExpression;
import slco.BooleanConstantExpression;
import slco.Channel;
import slco.ChannelTypeEnum;
import slco.ConstantExpression;
import slco.Delay;
import slco.Expression;
import slco.Final;
import slco.Initial;
import slco.IntegerConstantExpression;
import slco.Model;
import slco.OperatorEnum;
import slco.Port;
import slco.PrimitiveTypeEnum;
import slco.SendSignal;
import slco.SignalArgument;
import slco.SignalArgumentExpression;
import slco.SignalArgumentVariable;
import slco.SignalReception;
import slco.SlcoFactory;
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
 * An implementation of the model <b>Package</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class SlcoPackageImpl extends EPackageImpl implements SlcoPackage {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass argumentTypeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass assignmentEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass bidirectionalChannelEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass binaryOperatorExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass booleanConstantExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass channelEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass classEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass constantExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass delayEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass expressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass finalEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass initialEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass integerConstantExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass modelEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass objectEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass portEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass sendSignalEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass signalArgumentEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass signalArgumentExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass signalArgumentVariableEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass signalReceptionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass stateEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass stateMachineEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass statementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass stringConstantExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass textualStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass transitionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass unidirectionalChannelEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass variableEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass variableExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass vertexEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum channelTypeEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum operatorEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum primitiveTypeEnumEEnum = null;

	/**
	 * Creates an instance of the model <b>Package</b>, registered with
	 * {@link org.eclipse.emf.ecore.EPackage.Registry EPackage.Registry} by the package
	 * package URI value.
	 * <p>Note: the correct way to create the package is via the static
	 * factory method {@link #init init()}, which also performs
	 * initialization of the package, or returns the registered package,
	 * if one already exists.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see org.eclipse.emf.ecore.EPackage.Registry
	 * @see slco.SlcoPackage#eNS_URI
	 * @see #init()
	 * @generated
	 */
	private SlcoPackageImpl() {
		super(eNS_URI, SlcoFactory.eINSTANCE);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static boolean isInited = false;

	/**
	 * Creates, registers, and initializes the <b>Package</b> for this model, and for any others upon which it depends.
	 * 
	 * <p>This method is used to initialize {@link SlcoPackage#eINSTANCE} when that field is accessed.
	 * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #eNS_URI
	 * @see #createPackageContents()
	 * @see #initializePackageContents()
	 * @generated
	 */
	public static SlcoPackage init() {
		if (isInited) return (SlcoPackage)EPackage.Registry.INSTANCE.getEPackage(SlcoPackage.eNS_URI);

		// Obtain or create and register package
		SlcoPackageImpl theSlcoPackage = (SlcoPackageImpl)(EPackage.Registry.INSTANCE.get(eNS_URI) instanceof SlcoPackageImpl ? EPackage.Registry.INSTANCE.get(eNS_URI) : new SlcoPackageImpl());

		isInited = true;

		// Create package meta-data objects
		theSlcoPackage.createPackageContents();

		// Initialize created meta-data
		theSlcoPackage.initializePackageContents();

		// Mark meta-data to indicate it can't be changed
		theSlcoPackage.freeze();

  
		// Update the registry and return the package
		EPackage.Registry.INSTANCE.put(SlcoPackage.eNS_URI, theSlcoPackage);
		return theSlcoPackage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getArgumentType() {
		return argumentTypeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getArgumentType_Type() {
		return (EAttribute)argumentTypeEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getAssignment() {
		return assignmentEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAssignment_Variable() {
		return (EReference)assignmentEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAssignment_Expression() {
		return (EReference)assignmentEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBidirectionalChannel() {
		return bidirectionalChannelEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBidirectionalChannel_Port1() {
		return (EReference)bidirectionalChannelEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBidirectionalChannel_Port2() {
		return (EReference)bidirectionalChannelEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBidirectionalChannel_Object1() {
		return (EReference)bidirectionalChannelEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBidirectionalChannel_Object2() {
		return (EReference)bidirectionalChannelEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBinaryOperatorExpression() {
		return binaryOperatorExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getBinaryOperatorExpression_Operator() {
		return (EAttribute)binaryOperatorExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBinaryOperatorExpression_Operand1() {
		return (EReference)binaryOperatorExpressionEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBinaryOperatorExpression_Operand2() {
		return (EReference)binaryOperatorExpressionEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBooleanConstantExpression() {
		return booleanConstantExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getBooleanConstantExpression_Value() {
		return (EAttribute)booleanConstantExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getChannel() {
		return channelEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getChannel_ArgumentTypes() {
		return (EReference)channelEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getChannel_Name() {
		return (EAttribute)channelEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getChannel_ChannelType() {
		return (EAttribute)channelEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getClass_() {
		return classEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getClass_StateMachines() {
		return (EReference)classEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getClass_Ports() {
		return (EReference)classEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getClass_Variables() {
		return (EReference)classEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getClass_Name() {
		return (EAttribute)classEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getConstantExpression() {
		return constantExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getDelay() {
		return delayEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getDelay_Value() {
		return (EAttribute)delayEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getExpression() {
		return expressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getFinal() {
		return finalEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getInitial() {
		return initialEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getIntegerConstantExpression() {
		return integerConstantExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getIntegerConstantExpression_Value() {
		return (EAttribute)integerConstantExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getModel() {
		return modelEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getModel_Classes() {
		return (EReference)modelEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getModel_Objects() {
		return (EReference)modelEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getModel_Name() {
		return (EAttribute)modelEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getModel_Channels() {
		return (EReference)modelEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getObject() {
		return objectEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getObject_Name() {
		return (EAttribute)objectEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getObject_Class() {
		return (EReference)objectEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getPort() {
		return portEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getPort_Name() {
		return (EAttribute)portEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSendSignal() {
		return sendSignalEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSendSignal_Port() {
		return (EReference)sendSignalEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSendSignal_SignalName() {
		return (EAttribute)sendSignalEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSendSignal_Arguments() {
		return (EReference)sendSignalEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSignalArgument() {
		return signalArgumentEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSignalArgumentExpression() {
		return signalArgumentExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSignalArgumentExpression_Expression() {
		return (EReference)signalArgumentExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSignalArgumentVariable() {
		return signalArgumentVariableEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSignalArgumentVariable_Variable() {
		return (EReference)signalArgumentVariableEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSignalReception() {
		return signalReceptionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSignalReception_Port() {
		return (EReference)signalReceptionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSignalReception_SignalName() {
		return (EAttribute)signalReceptionEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSignalReception_Arguments() {
		return (EReference)signalReceptionEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSignalReception_Condition() {
		return (EReference)signalReceptionEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getState() {
		return stateEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getStateMachine() {
		return stateMachineEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getStateMachine_Name() {
		return (EAttribute)stateMachineEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getStateMachine_Vertices() {
		return (EReference)stateMachineEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getStateMachine_Transitions() {
		return (EReference)stateMachineEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getStateMachine_Variables() {
		return (EReference)stateMachineEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getStatement() {
		return statementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getStringConstantExpression() {
		return stringConstantExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getStringConstantExpression_Value() {
		return (EAttribute)stringConstantExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getTextualStatement() {
		return textualStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getTextualStatement_Statement() {
		return (EAttribute)textualStatementEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getTransition() {
		return transitionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getTransition_Name() {
		return (EAttribute)transitionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTransition_Source() {
		return (EReference)transitionEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTransition_Target() {
		return (EReference)transitionEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTransition_Statements() {
		return (EReference)transitionEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getUnidirectionalChannel() {
		return unidirectionalChannelEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getUnidirectionalChannel_SourcePort() {
		return (EReference)unidirectionalChannelEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getUnidirectionalChannel_TargetPort() {
		return (EReference)unidirectionalChannelEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getUnidirectionalChannel_SourceObject() {
		return (EReference)unidirectionalChannelEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getUnidirectionalChannel_TargetObject() {
		return (EReference)unidirectionalChannelEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getVariable() {
		return variableEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getVariable_Name() {
		return (EAttribute)variableEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getVariable_InitialValue() {
		return (EReference)variableEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getVariable_Type() {
		return (EAttribute)variableEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getVariableExpression() {
		return variableExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getVariableExpression_Variable() {
		return (EReference)variableExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getVertex() {
		return vertexEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getVertex_Name() {
		return (EAttribute)vertexEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getVertex_Incoming() {
		return (EReference)vertexEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getVertex_Outgoing() {
		return (EReference)vertexEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getChannelTypeEnum() {
		return channelTypeEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getOperatorEnum() {
		return operatorEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getPrimitiveTypeEnum() {
		return primitiveTypeEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SlcoFactory getSlcoFactory() {
		return (SlcoFactory)getEFactoryInstance();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private boolean isCreated = false;

	/**
	 * Creates the meta-model objects for the package.  This method is
	 * guarded to have no affect on any invocation but its first.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void createPackageContents() {
		if (isCreated) return;
		isCreated = true;

		// Create classes and their features
		argumentTypeEClass = createEClass(ARGUMENT_TYPE);
		createEAttribute(argumentTypeEClass, ARGUMENT_TYPE__TYPE);

		assignmentEClass = createEClass(ASSIGNMENT);
		createEReference(assignmentEClass, ASSIGNMENT__VARIABLE);
		createEReference(assignmentEClass, ASSIGNMENT__EXPRESSION);

		bidirectionalChannelEClass = createEClass(BIDIRECTIONAL_CHANNEL);
		createEReference(bidirectionalChannelEClass, BIDIRECTIONAL_CHANNEL__PORT1);
		createEReference(bidirectionalChannelEClass, BIDIRECTIONAL_CHANNEL__PORT2);
		createEReference(bidirectionalChannelEClass, BIDIRECTIONAL_CHANNEL__OBJECT1);
		createEReference(bidirectionalChannelEClass, BIDIRECTIONAL_CHANNEL__OBJECT2);

		binaryOperatorExpressionEClass = createEClass(BINARY_OPERATOR_EXPRESSION);
		createEAttribute(binaryOperatorExpressionEClass, BINARY_OPERATOR_EXPRESSION__OPERATOR);
		createEReference(binaryOperatorExpressionEClass, BINARY_OPERATOR_EXPRESSION__OPERAND1);
		createEReference(binaryOperatorExpressionEClass, BINARY_OPERATOR_EXPRESSION__OPERAND2);

		booleanConstantExpressionEClass = createEClass(BOOLEAN_CONSTANT_EXPRESSION);
		createEAttribute(booleanConstantExpressionEClass, BOOLEAN_CONSTANT_EXPRESSION__VALUE);

		channelEClass = createEClass(CHANNEL);
		createEReference(channelEClass, CHANNEL__ARGUMENT_TYPES);
		createEAttribute(channelEClass, CHANNEL__NAME);
		createEAttribute(channelEClass, CHANNEL__CHANNEL_TYPE);

		classEClass = createEClass(CLASS);
		createEReference(classEClass, CLASS__STATE_MACHINES);
		createEReference(classEClass, CLASS__PORTS);
		createEReference(classEClass, CLASS__VARIABLES);
		createEAttribute(classEClass, CLASS__NAME);

		constantExpressionEClass = createEClass(CONSTANT_EXPRESSION);

		delayEClass = createEClass(DELAY);
		createEAttribute(delayEClass, DELAY__VALUE);

		expressionEClass = createEClass(EXPRESSION);

		finalEClass = createEClass(FINAL);

		initialEClass = createEClass(INITIAL);

		integerConstantExpressionEClass = createEClass(INTEGER_CONSTANT_EXPRESSION);
		createEAttribute(integerConstantExpressionEClass, INTEGER_CONSTANT_EXPRESSION__VALUE);

		modelEClass = createEClass(MODEL);
		createEReference(modelEClass, MODEL__CLASSES);
		createEReference(modelEClass, MODEL__OBJECTS);
		createEAttribute(modelEClass, MODEL__NAME);
		createEReference(modelEClass, MODEL__CHANNELS);

		objectEClass = createEClass(OBJECT);
		createEAttribute(objectEClass, OBJECT__NAME);
		createEReference(objectEClass, OBJECT__CLASS);

		portEClass = createEClass(PORT);
		createEAttribute(portEClass, PORT__NAME);

		sendSignalEClass = createEClass(SEND_SIGNAL);
		createEReference(sendSignalEClass, SEND_SIGNAL__PORT);
		createEAttribute(sendSignalEClass, SEND_SIGNAL__SIGNAL_NAME);
		createEReference(sendSignalEClass, SEND_SIGNAL__ARGUMENTS);

		signalArgumentEClass = createEClass(SIGNAL_ARGUMENT);

		signalArgumentExpressionEClass = createEClass(SIGNAL_ARGUMENT_EXPRESSION);
		createEReference(signalArgumentExpressionEClass, SIGNAL_ARGUMENT_EXPRESSION__EXPRESSION);

		signalArgumentVariableEClass = createEClass(SIGNAL_ARGUMENT_VARIABLE);
		createEReference(signalArgumentVariableEClass, SIGNAL_ARGUMENT_VARIABLE__VARIABLE);

		signalReceptionEClass = createEClass(SIGNAL_RECEPTION);
		createEReference(signalReceptionEClass, SIGNAL_RECEPTION__PORT);
		createEAttribute(signalReceptionEClass, SIGNAL_RECEPTION__SIGNAL_NAME);
		createEReference(signalReceptionEClass, SIGNAL_RECEPTION__ARGUMENTS);
		createEReference(signalReceptionEClass, SIGNAL_RECEPTION__CONDITION);

		stateEClass = createEClass(STATE);

		stateMachineEClass = createEClass(STATE_MACHINE);
		createEAttribute(stateMachineEClass, STATE_MACHINE__NAME);
		createEReference(stateMachineEClass, STATE_MACHINE__VERTICES);
		createEReference(stateMachineEClass, STATE_MACHINE__TRANSITIONS);
		createEReference(stateMachineEClass, STATE_MACHINE__VARIABLES);

		statementEClass = createEClass(STATEMENT);

		stringConstantExpressionEClass = createEClass(STRING_CONSTANT_EXPRESSION);
		createEAttribute(stringConstantExpressionEClass, STRING_CONSTANT_EXPRESSION__VALUE);

		textualStatementEClass = createEClass(TEXTUAL_STATEMENT);
		createEAttribute(textualStatementEClass, TEXTUAL_STATEMENT__STATEMENT);

		transitionEClass = createEClass(TRANSITION);
		createEAttribute(transitionEClass, TRANSITION__NAME);
		createEReference(transitionEClass, TRANSITION__SOURCE);
		createEReference(transitionEClass, TRANSITION__TARGET);
		createEReference(transitionEClass, TRANSITION__STATEMENTS);

		unidirectionalChannelEClass = createEClass(UNIDIRECTIONAL_CHANNEL);
		createEReference(unidirectionalChannelEClass, UNIDIRECTIONAL_CHANNEL__SOURCE_PORT);
		createEReference(unidirectionalChannelEClass, UNIDIRECTIONAL_CHANNEL__TARGET_PORT);
		createEReference(unidirectionalChannelEClass, UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT);
		createEReference(unidirectionalChannelEClass, UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT);

		variableEClass = createEClass(VARIABLE);
		createEAttribute(variableEClass, VARIABLE__NAME);
		createEReference(variableEClass, VARIABLE__INITIAL_VALUE);
		createEAttribute(variableEClass, VARIABLE__TYPE);

		variableExpressionEClass = createEClass(VARIABLE_EXPRESSION);
		createEReference(variableExpressionEClass, VARIABLE_EXPRESSION__VARIABLE);

		vertexEClass = createEClass(VERTEX);
		createEAttribute(vertexEClass, VERTEX__NAME);
		createEReference(vertexEClass, VERTEX__INCOMING);
		createEReference(vertexEClass, VERTEX__OUTGOING);

		// Create enums
		channelTypeEnumEEnum = createEEnum(CHANNEL_TYPE_ENUM);
		operatorEnumEEnum = createEEnum(OPERATOR_ENUM);
		primitiveTypeEnumEEnum = createEEnum(PRIMITIVE_TYPE_ENUM);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private boolean isInitialized = false;

	/**
	 * Complete the initialization of the package and its meta-model.  This
	 * method is guarded to have no affect on any invocation but its first.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void initializePackageContents() {
		if (isInitialized) return;
		isInitialized = true;

		// Initialize package
		setName(eNAME);
		setNsPrefix(eNS_PREFIX);
		setNsURI(eNS_URI);

		// Create type parameters

		// Set bounds for type parameters

		// Add supertypes to classes
		assignmentEClass.getESuperTypes().add(this.getStatement());
		bidirectionalChannelEClass.getESuperTypes().add(this.getChannel());
		binaryOperatorExpressionEClass.getESuperTypes().add(this.getExpression());
		booleanConstantExpressionEClass.getESuperTypes().add(this.getConstantExpression());
		constantExpressionEClass.getESuperTypes().add(this.getExpression());
		delayEClass.getESuperTypes().add(this.getStatement());
		expressionEClass.getESuperTypes().add(this.getStatement());
		finalEClass.getESuperTypes().add(this.getVertex());
		initialEClass.getESuperTypes().add(this.getVertex());
		integerConstantExpressionEClass.getESuperTypes().add(this.getConstantExpression());
		sendSignalEClass.getESuperTypes().add(this.getStatement());
		signalArgumentExpressionEClass.getESuperTypes().add(this.getSignalArgument());
		signalArgumentVariableEClass.getESuperTypes().add(this.getSignalArgument());
		signalReceptionEClass.getESuperTypes().add(this.getStatement());
		stateEClass.getESuperTypes().add(this.getVertex());
		stringConstantExpressionEClass.getESuperTypes().add(this.getConstantExpression());
		textualStatementEClass.getESuperTypes().add(this.getStatement());
		unidirectionalChannelEClass.getESuperTypes().add(this.getChannel());
		variableExpressionEClass.getESuperTypes().add(this.getExpression());

		// Initialize classes and features; add operations and parameters
		initEClass(argumentTypeEClass, ArgumentType.class, "ArgumentType", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getArgumentType_Type(), this.getPrimitiveTypeEnum(), "type", null, 1, 1, ArgumentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(assignmentEClass, Assignment.class, "Assignment", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getAssignment_Variable(), this.getVariable(), null, "variable", null, 1, 1, Assignment.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getAssignment_Expression(), this.getExpression(), null, "expression", null, 1, 1, Assignment.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, !IS_ORDERED);

		initEClass(bidirectionalChannelEClass, BidirectionalChannel.class, "BidirectionalChannel", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getBidirectionalChannel_Port1(), this.getPort(), null, "port1", null, 1, 1, BidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getBidirectionalChannel_Port2(), this.getPort(), null, "port2", null, 1, 1, BidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getBidirectionalChannel_Object1(), this.getObject(), null, "object1", null, 1, 1, BidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getBidirectionalChannel_Object2(), this.getObject(), null, "object2", null, 1, 1, BidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(binaryOperatorExpressionEClass, BinaryOperatorExpression.class, "BinaryOperatorExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getBinaryOperatorExpression_Operator(), this.getOperatorEnum(), "operator", null, 1, 1, BinaryOperatorExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getBinaryOperatorExpression_Operand1(), this.getExpression(), null, "operand1", null, 1, 1, BinaryOperatorExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, !IS_ORDERED);
		initEReference(getBinaryOperatorExpression_Operand2(), this.getExpression(), null, "operand2", null, 1, 1, BinaryOperatorExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, !IS_ORDERED);

		initEClass(booleanConstantExpressionEClass, BooleanConstantExpression.class, "BooleanConstantExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getBooleanConstantExpression_Value(), ecorePackage.getEBoolean(), "value", null, 1, 1, BooleanConstantExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, !IS_ORDERED);

		initEClass(channelEClass, Channel.class, "Channel", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getChannel_ArgumentTypes(), this.getArgumentType(), null, "argumentTypes", null, 0, -1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getChannel_Name(), ecorePackage.getEString(), "name", null, 1, 1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getChannel_ChannelType(), this.getChannelTypeEnum(), "channelType", null, 1, 1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(classEClass, slco.Class.class, "Class", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getClass_StateMachines(), this.getStateMachine(), null, "stateMachines", null, 0, -1, slco.Class.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getClass_Ports(), this.getPort(), null, "ports", null, 0, -1, slco.Class.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getClass_Variables(), this.getVariable(), null, "variables", null, 0, -1, slco.Class.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getClass_Name(), ecorePackage.getEString(), "name", null, 1, 1, slco.Class.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(constantExpressionEClass, ConstantExpression.class, "ConstantExpression", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(delayEClass, Delay.class, "Delay", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getDelay_Value(), ecorePackage.getEInt(), "value", "0", 1, 1, Delay.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(expressionEClass, Expression.class, "Expression", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(finalEClass, Final.class, "Final", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(initialEClass, Initial.class, "Initial", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(integerConstantExpressionEClass, IntegerConstantExpression.class, "IntegerConstantExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getIntegerConstantExpression_Value(), ecorePackage.getEInt(), "value", null, 1, 1, IntegerConstantExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, !IS_UNIQUE, !IS_DERIVED, !IS_ORDERED);

		initEClass(modelEClass, Model.class, "Model", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getModel_Classes(), this.getClass_(), null, "classes", null, 0, -1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getModel_Objects(), this.getObject(), null, "objects", null, 0, -1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getModel_Name(), ecorePackage.getEString(), "name", null, 1, 1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getModel_Channels(), this.getChannel(), null, "channels", null, 0, -1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(objectEClass, slco.Object.class, "Object", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getObject_Name(), ecorePackage.getEString(), "name", null, 1, 1, slco.Object.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getObject_Class(), this.getClass_(), null, "class", null, 1, 1, slco.Object.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(portEClass, Port.class, "Port", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getPort_Name(), ecorePackage.getEString(), "name", null, 1, 1, Port.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(sendSignalEClass, SendSignal.class, "SendSignal", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getSendSignal_Port(), this.getPort(), null, "port", null, 1, 1, SendSignal.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, !IS_ORDERED);
		initEAttribute(getSendSignal_SignalName(), ecorePackage.getEString(), "signalName", null, 1, 1, SendSignal.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getSendSignal_Arguments(), this.getExpression(), null, "arguments", null, 0, -1, SendSignal.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(signalArgumentEClass, SignalArgument.class, "SignalArgument", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(signalArgumentExpressionEClass, SignalArgumentExpression.class, "SignalArgumentExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getSignalArgumentExpression_Expression(), this.getExpression(), null, "expression", null, 0, 1, SignalArgumentExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(signalArgumentVariableEClass, SignalArgumentVariable.class, "SignalArgumentVariable", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getSignalArgumentVariable_Variable(), this.getVariable(), null, "variable", null, 0, 1, SignalArgumentVariable.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(signalReceptionEClass, SignalReception.class, "SignalReception", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getSignalReception_Port(), this.getPort(), null, "port", null, 1, 1, SignalReception.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getSignalReception_SignalName(), ecorePackage.getEString(), "signalName", null, 1, 1, SignalReception.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getSignalReception_Arguments(), this.getSignalArgument(), null, "arguments", null, 0, -1, SignalReception.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getSignalReception_Condition(), this.getExpression(), null, "condition", null, 0, 1, SignalReception.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(stateEClass, State.class, "State", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(stateMachineEClass, StateMachine.class, "StateMachine", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getStateMachine_Name(), ecorePackage.getEString(), "name", null, 1, 1, StateMachine.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getStateMachine_Vertices(), this.getVertex(), null, "vertices", null, 0, -1, StateMachine.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getStateMachine_Transitions(), this.getTransition(), null, "transitions", null, 0, -1, StateMachine.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getStateMachine_Variables(), this.getVariable(), null, "variables", null, 0, -1, StateMachine.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(statementEClass, Statement.class, "Statement", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(stringConstantExpressionEClass, StringConstantExpression.class, "StringConstantExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getStringConstantExpression_Value(), ecorePackage.getEString(), "value", null, 1, 1, StringConstantExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(textualStatementEClass, TextualStatement.class, "TextualStatement", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getTextualStatement_Statement(), ecorePackage.getEString(), "statement", null, 1, 1, TextualStatement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(transitionEClass, Transition.class, "Transition", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getTransition_Name(), ecorePackage.getEString(), "name", null, 1, 1, Transition.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getTransition_Source(), this.getVertex(), this.getVertex_Outgoing(), "source", null, 1, 1, Transition.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getTransition_Target(), this.getVertex(), this.getVertex_Incoming(), "target", null, 1, 1, Transition.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getTransition_Statements(), this.getStatement(), null, "statements", null, 0, -1, Transition.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(unidirectionalChannelEClass, UnidirectionalChannel.class, "UnidirectionalChannel", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getUnidirectionalChannel_SourcePort(), this.getPort(), null, "sourcePort", null, 1, 1, UnidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getUnidirectionalChannel_TargetPort(), this.getPort(), null, "targetPort", null, 1, 1, UnidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getUnidirectionalChannel_SourceObject(), this.getObject(), null, "sourceObject", null, 1, 1, UnidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getUnidirectionalChannel_TargetObject(), this.getObject(), null, "targetObject", null, 1, 1, UnidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(variableEClass, Variable.class, "Variable", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getVariable_Name(), ecorePackage.getEString(), "name", null, 1, 1, Variable.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getVariable_InitialValue(), this.getExpression(), null, "initialValue", null, 0, 1, Variable.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getVariable_Type(), this.getPrimitiveTypeEnum(), "type", null, 1, 1, Variable.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(variableExpressionEClass, VariableExpression.class, "VariableExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getVariableExpression_Variable(), this.getVariable(), null, "variable", null, 1, 1, VariableExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(vertexEClass, Vertex.class, "Vertex", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getVertex_Name(), ecorePackage.getEString(), "name", null, 1, 1, Vertex.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getVertex_Incoming(), this.getTransition(), this.getTransition_Target(), "incoming", null, 0, -1, Vertex.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getVertex_Outgoing(), this.getTransition(), this.getTransition_Source(), "outgoing", null, 0, -1, Vertex.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		// Initialize enums and add enum literals
		initEEnum(channelTypeEnumEEnum, ChannelTypeEnum.class, "ChannelTypeEnum");
		addEEnumLiteral(channelTypeEnumEEnum, ChannelTypeEnum.ASYNC_LOSSLESS);
		addEEnumLiteral(channelTypeEnumEEnum, ChannelTypeEnum.ASYNC_LOSSY);
		addEEnumLiteral(channelTypeEnumEEnum, ChannelTypeEnum.SYNC);

		initEEnum(operatorEnumEEnum, OperatorEnum.class, "OperatorEnum");
		addEEnumLiteral(operatorEnumEEnum, OperatorEnum.AT_LEAST);
		addEEnumLiteral(operatorEnumEEnum, OperatorEnum.ADD);
		addEEnumLiteral(operatorEnumEEnum, OperatorEnum.AND);
		addEEnumLiteral(operatorEnumEEnum, OperatorEnum.EQUALS);
		addEEnumLiteral(operatorEnumEEnum, OperatorEnum.DIFFERS);
		addEEnumLiteral(operatorEnumEEnum, OperatorEnum.SUBTRACT);
		addEEnumLiteral(operatorEnumEEnum, OperatorEnum.AT_MOST);
		addEEnumLiteral(operatorEnumEEnum, OperatorEnum.OR);

		initEEnum(primitiveTypeEnumEEnum, PrimitiveTypeEnum.class, "PrimitiveTypeEnum");
		addEEnumLiteral(primitiveTypeEnumEEnum, PrimitiveTypeEnum.INTEGER);
		addEEnumLiteral(primitiveTypeEnumEEnum, PrimitiveTypeEnum.BOOLEAN);
		addEEnumLiteral(primitiveTypeEnumEEnum, PrimitiveTypeEnum.STRING);

		// Create resource
		createResource(eNS_URI);
	}

} //SlcoPackageImpl
