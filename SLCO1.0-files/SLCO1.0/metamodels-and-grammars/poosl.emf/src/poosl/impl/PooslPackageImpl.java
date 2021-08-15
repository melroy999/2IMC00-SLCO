/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

import org.eclipse.emf.ecore.impl.EPackageImpl;

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
import poosl.Type;
import poosl.Variable;
import poosl.VariableExpression;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Package</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class PooslPackageImpl extends EPackageImpl implements PooslPackage {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass abortEClass = null;

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
	private EClass atomicEClass = null;

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
	private EClass conditionalExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass conditionalStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass constantEClass = null;

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
	private EClass currentTimeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass dataClassEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass dataMethodEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass dataMethodCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass primitiveDataMethodCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass dataObjectCreationEClass = null;

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
	private EClass expressionListEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass guardedExecutionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass incomingMessageEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass interruptEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass loopExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass loopStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass messageEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass messageReceiveEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass messageSendEClass = null;

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
	private EClass outgoingMessageEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass parallelCompositionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass parameterEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass parameterExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass primitiveTypeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass processClassEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass processInstanceEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass processMethodEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass processMethodCallEClass = null;

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
	private EClass referenceSelfEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass returnExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass selectionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass skipEClass = null;

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
	private EClass statementListEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass timeSynchronisationEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass typeEClass = null;

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
	private EEnum primitiveTypeEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum primitiveDataMethodEnumEEnum = null;

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
	 * @see poosl.PooslPackage#eNS_URI
	 * @see #init()
	 * @generated
	 */
	private PooslPackageImpl() {
		super(eNS_URI, PooslFactory.eINSTANCE);
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
	 * <p>This method is used to initialize {@link PooslPackage#eINSTANCE} when that field is accessed.
	 * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #eNS_URI
	 * @see #createPackageContents()
	 * @see #initializePackageContents()
	 * @generated
	 */
	public static PooslPackage init() {
		if (isInited) return (PooslPackage)EPackage.Registry.INSTANCE.getEPackage(PooslPackage.eNS_URI);

		// Obtain or create and register package
		PooslPackageImpl thePooslPackage = (PooslPackageImpl)(EPackage.Registry.INSTANCE.get(eNS_URI) instanceof PooslPackageImpl ? EPackage.Registry.INSTANCE.get(eNS_URI) : new PooslPackageImpl());

		isInited = true;

		// Create package meta-data objects
		thePooslPackage.createPackageContents();

		// Initialize created meta-data
		thePooslPackage.initializePackageContents();

		// Mark meta-data to indicate it can't be changed
		thePooslPackage.freeze();

  
		// Update the registry and return the package
		EPackage.Registry.INSTANCE.put(PooslPackage.eNS_URI, thePooslPackage);
		return thePooslPackage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getAbort() {
		return abortEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAbort_Body() {
		return (EReference)abortEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAbort_Condition() {
		return (EReference)abortEClass.getEStructuralFeatures().get(1);
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
	public EReference getAssignment_RightHandSide() {
		return (EReference)assignmentEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAssignment_LeftHandSide() {
		return (EReference)assignmentEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getAtomic() {
		return atomicEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAtomic_Statements() {
		return (EReference)atomicEClass.getEStructuralFeatures().get(0);
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
	public EAttribute getChannel_Name() {
		return (EAttribute)channelEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getChannel_Port1() {
		return (EReference)channelEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getChannel_Port2() {
		return (EReference)channelEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getChannel_ProcessInstance1() {
		return (EReference)channelEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getChannel_ProcessInstance2() {
		return (EReference)channelEClass.getEStructuralFeatures().get(4);
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
	public EAttribute getClass_Name() {
		return (EAttribute)classEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getConditionalExpression() {
		return conditionalExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getConditionalExpression_Condition() {
		return (EReference)conditionalExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getConditionalExpression_ThenClause() {
		return (EReference)conditionalExpressionEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getConditionalExpression_ElseClause() {
		return (EReference)conditionalExpressionEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getConditionalStatement() {
		return conditionalStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getConditionalStatement_Condition() {
		return (EReference)conditionalStatementEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getConditionalStatement_ThenClause() {
		return (EReference)conditionalStatementEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getConditionalStatement_ElseClause() {
		return (EReference)conditionalStatementEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getConstant() {
		return constantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getConstant_Integer() {
		return (EAttribute)constantEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getConstant_String() {
		return (EAttribute)constantEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getConstant_Boolean() {
		return (EAttribute)constantEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getConstant_Real() {
		return (EAttribute)constantEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getConstant_Type() {
		return (EAttribute)constantEClass.getEStructuralFeatures().get(4);
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
	public EReference getConstantExpression_Constant() {
		return (EReference)constantExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCurrentTime() {
		return currentTimeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getDataClass() {
		return dataClassEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataClass_Variables() {
		return (EReference)dataClassEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataClass_Methods() {
		return (EReference)dataClassEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataClass_SuperClass() {
		return (EReference)dataClassEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getDataMethod() {
		return dataMethodEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataMethod_Parameters() {
		return (EReference)dataMethodEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getDataMethod_Name() {
		return (EAttribute)dataMethodEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataMethod_Variables() {
		return (EReference)dataMethodEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataMethod_Body() {
		return (EReference)dataMethodEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataMethod_ReturnType() {
		return (EReference)dataMethodEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getDataMethodCall() {
		return dataMethodCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getDataMethodCall_Super() {
		return (EAttribute)dataMethodCallEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataMethodCall_ObjectExpressions() {
		return (EReference)dataMethodCallEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataMethodCall_ParameterExpressionLists() {
		return (EReference)dataMethodCallEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataMethodCall_DataMethod() {
		return (EReference)dataMethodCallEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getPrimitiveDataMethodCall() {
		return primitiveDataMethodCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getPrimitiveDataMethodCall_ObjectExpressions() {
		return (EReference)primitiveDataMethodCallEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getPrimitiveDataMethodCall_ParameterExpressionLists() {
		return (EReference)primitiveDataMethodCallEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getPrimitiveDataMethodCall_PrimitiveDataMethod() {
		return (EAttribute)primitiveDataMethodCallEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getDataObjectCreation() {
		return dataObjectCreationEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDataObjectCreation_DataClass() {
		return (EReference)dataObjectCreationEClass.getEStructuralFeatures().get(0);
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
	public EClass getExpressionList() {
		return expressionListEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getExpressionList_Expressions() {
		return (EReference)expressionListEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getGuardedExecution() {
		return guardedExecutionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getGuardedExecution_Statement() {
		return (EReference)guardedExecutionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getGuardedExecution_Guard() {
		return (EReference)guardedExecutionEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getIncomingMessage() {
		return incomingMessageEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getInterrupt() {
		return interruptEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getInterrupt_Body() {
		return (EReference)interruptEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getInterrupt_Condition() {
		return (EReference)interruptEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getLoopExpression() {
		return loopExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getLoopExpression_Condition() {
		return (EReference)loopExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getLoopExpression_Body() {
		return (EReference)loopExpressionEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getLoopStatement() {
		return loopStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getLoopStatement_Body() {
		return (EReference)loopStatementEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getLoopStatement_Condition() {
		return (EReference)loopStatementEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getMessage() {
		return messageEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getMessage_Name() {
		return (EAttribute)messageEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMessage_ParameterTypes() {
		return (EReference)messageEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getMessageReceive() {
		return messageReceiveEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMessageReceive_Message() {
		return (EReference)messageReceiveEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMessageReceive_Port() {
		return (EReference)messageReceiveEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMessageReceive_Variables() {
		return (EReference)messageReceiveEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMessageReceive_Condition() {
		return (EReference)messageReceiveEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMessageReceive_Expressions() {
		return (EReference)messageReceiveEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getMessageSend() {
		return messageSendEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMessageSend_Message() {
		return (EReference)messageSendEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMessageSend_Port() {
		return (EReference)messageSendEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMessageSend_Expressions() {
		return (EReference)messageSendEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMessageSend_ExpressionLists() {
		return (EReference)messageSendEClass.getEStructuralFeatures().get(3);
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
	public EAttribute getModel_Name() {
		return (EAttribute)modelEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getModel_Classes() {
		return (EReference)modelEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getModel_ProcessInstances() {
		return (EReference)modelEClass.getEStructuralFeatures().get(2);
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
	public EReference getModel_PrimitiveTypes() {
		return (EReference)modelEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getOutgoingMessage() {
		return outgoingMessageEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getParallelComposition() {
		return parallelCompositionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getParallelComposition_StatementLists() {
		return (EReference)parallelCompositionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getParameter() {
		return parameterEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getParameter_Type() {
		return (EReference)parameterEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getParameter_Name() {
		return (EAttribute)parameterEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getParameterExpression() {
		return parameterExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getParameterExpression_Parameter() {
		return (EReference)parameterExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getPrimitiveType() {
		return primitiveTypeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getPrimitiveType_PrimitiveTypeEnum() {
		return (EAttribute)primitiveTypeEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getProcessClass() {
		return processClassEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessClass_Variables() {
		return (EReference)processClassEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessClass_Methods() {
		return (EReference)processClassEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessClass_SuperClass() {
		return (EReference)processClassEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessClass_Parameters() {
		return (EReference)processClassEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessClass_Ports() {
		return (EReference)processClassEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessClass_InitialMethodCall() {
		return (EReference)processClassEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getProcessInstance() {
		return processInstanceEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getProcessInstance_Name() {
		return (EAttribute)processInstanceEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessInstance_Class() {
		return (EReference)processInstanceEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getProcessMethod() {
		return processMethodEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessMethod_InputParameters() {
		return (EReference)processMethodEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessMethod_OutputParameters() {
		return (EReference)processMethodEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessMethod_Variables() {
		return (EReference)processMethodEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getProcessMethod_Name() {
		return (EAttribute)processMethodEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessMethod_Body() {
		return (EReference)processMethodEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getProcessMethodCall() {
		return processMethodCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessMethodCall_Method() {
		return (EReference)processMethodCallEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessMethodCall_ParameterExpressions() {
		return (EReference)processMethodCallEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProcessMethodCall_Variables() {
		return (EReference)processMethodCallEClass.getEStructuralFeatures().get(2);
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
	public EClass getReferenceSelf() {
		return referenceSelfEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getReturnExpression() {
		return returnExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getReturnExpression_ReturnExpressions() {
		return (EReference)returnExpressionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSelection() {
		return selectionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSelection_StatementLists() {
		return (EReference)selectionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSkip() {
		return skipEClass;
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
	public EClass getStatementList() {
		return statementListEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getStatementList_Statements() {
		return (EReference)statementListEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getTimeSynchronisation() {
		return timeSynchronisationEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTimeSynchronisation_Duration() {
		return (EReference)timeSynchronisationEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getType() {
		return typeEClass;
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
	public EReference getVariable_Type() {
		return (EReference)variableEClass.getEStructuralFeatures().get(1);
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
	public EEnum getPrimitiveTypeEnum() {
		return primitiveTypeEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getPrimitiveDataMethodEnum() {
		return primitiveDataMethodEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PooslFactory getPooslFactory() {
		return (PooslFactory)getEFactoryInstance();
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
		abortEClass = createEClass(ABORT);
		createEReference(abortEClass, ABORT__BODY);
		createEReference(abortEClass, ABORT__CONDITION);

		assignmentEClass = createEClass(ASSIGNMENT);
		createEReference(assignmentEClass, ASSIGNMENT__RIGHT_HAND_SIDE);
		createEReference(assignmentEClass, ASSIGNMENT__LEFT_HAND_SIDE);

		atomicEClass = createEClass(ATOMIC);
		createEReference(atomicEClass, ATOMIC__STATEMENTS);

		channelEClass = createEClass(CHANNEL);
		createEAttribute(channelEClass, CHANNEL__NAME);
		createEReference(channelEClass, CHANNEL__PORT1);
		createEReference(channelEClass, CHANNEL__PORT2);
		createEReference(channelEClass, CHANNEL__PROCESS_INSTANCE1);
		createEReference(channelEClass, CHANNEL__PROCESS_INSTANCE2);

		classEClass = createEClass(CLASS);
		createEAttribute(classEClass, CLASS__NAME);

		conditionalExpressionEClass = createEClass(CONDITIONAL_EXPRESSION);
		createEReference(conditionalExpressionEClass, CONDITIONAL_EXPRESSION__CONDITION);
		createEReference(conditionalExpressionEClass, CONDITIONAL_EXPRESSION__THEN_CLAUSE);
		createEReference(conditionalExpressionEClass, CONDITIONAL_EXPRESSION__ELSE_CLAUSE);

		conditionalStatementEClass = createEClass(CONDITIONAL_STATEMENT);
		createEReference(conditionalStatementEClass, CONDITIONAL_STATEMENT__CONDITION);
		createEReference(conditionalStatementEClass, CONDITIONAL_STATEMENT__THEN_CLAUSE);
		createEReference(conditionalStatementEClass, CONDITIONAL_STATEMENT__ELSE_CLAUSE);

		constantEClass = createEClass(CONSTANT);
		createEAttribute(constantEClass, CONSTANT__INTEGER);
		createEAttribute(constantEClass, CONSTANT__STRING);
		createEAttribute(constantEClass, CONSTANT__BOOLEAN);
		createEAttribute(constantEClass, CONSTANT__REAL);
		createEAttribute(constantEClass, CONSTANT__TYPE);

		constantExpressionEClass = createEClass(CONSTANT_EXPRESSION);
		createEReference(constantExpressionEClass, CONSTANT_EXPRESSION__CONSTANT);

		currentTimeEClass = createEClass(CURRENT_TIME);

		dataClassEClass = createEClass(DATA_CLASS);
		createEReference(dataClassEClass, DATA_CLASS__VARIABLES);
		createEReference(dataClassEClass, DATA_CLASS__METHODS);
		createEReference(dataClassEClass, DATA_CLASS__SUPER_CLASS);

		dataMethodEClass = createEClass(DATA_METHOD);
		createEReference(dataMethodEClass, DATA_METHOD__PARAMETERS);
		createEAttribute(dataMethodEClass, DATA_METHOD__NAME);
		createEReference(dataMethodEClass, DATA_METHOD__VARIABLES);
		createEReference(dataMethodEClass, DATA_METHOD__BODY);
		createEReference(dataMethodEClass, DATA_METHOD__RETURN_TYPE);

		dataMethodCallEClass = createEClass(DATA_METHOD_CALL);
		createEAttribute(dataMethodCallEClass, DATA_METHOD_CALL__SUPER);
		createEReference(dataMethodCallEClass, DATA_METHOD_CALL__OBJECT_EXPRESSIONS);
		createEReference(dataMethodCallEClass, DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS);
		createEReference(dataMethodCallEClass, DATA_METHOD_CALL__DATA_METHOD);

		primitiveDataMethodCallEClass = createEClass(PRIMITIVE_DATA_METHOD_CALL);
		createEReference(primitiveDataMethodCallEClass, PRIMITIVE_DATA_METHOD_CALL__OBJECT_EXPRESSIONS);
		createEReference(primitiveDataMethodCallEClass, PRIMITIVE_DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS);
		createEAttribute(primitiveDataMethodCallEClass, PRIMITIVE_DATA_METHOD_CALL__PRIMITIVE_DATA_METHOD);

		dataObjectCreationEClass = createEClass(DATA_OBJECT_CREATION);
		createEReference(dataObjectCreationEClass, DATA_OBJECT_CREATION__DATA_CLASS);

		expressionEClass = createEClass(EXPRESSION);

		expressionListEClass = createEClass(EXPRESSION_LIST);
		createEReference(expressionListEClass, EXPRESSION_LIST__EXPRESSIONS);

		guardedExecutionEClass = createEClass(GUARDED_EXECUTION);
		createEReference(guardedExecutionEClass, GUARDED_EXECUTION__STATEMENT);
		createEReference(guardedExecutionEClass, GUARDED_EXECUTION__GUARD);

		incomingMessageEClass = createEClass(INCOMING_MESSAGE);

		interruptEClass = createEClass(INTERRUPT);
		createEReference(interruptEClass, INTERRUPT__BODY);
		createEReference(interruptEClass, INTERRUPT__CONDITION);

		loopExpressionEClass = createEClass(LOOP_EXPRESSION);
		createEReference(loopExpressionEClass, LOOP_EXPRESSION__CONDITION);
		createEReference(loopExpressionEClass, LOOP_EXPRESSION__BODY);

		loopStatementEClass = createEClass(LOOP_STATEMENT);
		createEReference(loopStatementEClass, LOOP_STATEMENT__BODY);
		createEReference(loopStatementEClass, LOOP_STATEMENT__CONDITION);

		messageEClass = createEClass(MESSAGE);
		createEAttribute(messageEClass, MESSAGE__NAME);
		createEReference(messageEClass, MESSAGE__PARAMETER_TYPES);

		messageReceiveEClass = createEClass(MESSAGE_RECEIVE);
		createEReference(messageReceiveEClass, MESSAGE_RECEIVE__MESSAGE);
		createEReference(messageReceiveEClass, MESSAGE_RECEIVE__PORT);
		createEReference(messageReceiveEClass, MESSAGE_RECEIVE__VARIABLES);
		createEReference(messageReceiveEClass, MESSAGE_RECEIVE__CONDITION);
		createEReference(messageReceiveEClass, MESSAGE_RECEIVE__EXPRESSIONS);

		messageSendEClass = createEClass(MESSAGE_SEND);
		createEReference(messageSendEClass, MESSAGE_SEND__MESSAGE);
		createEReference(messageSendEClass, MESSAGE_SEND__PORT);
		createEReference(messageSendEClass, MESSAGE_SEND__EXPRESSIONS);
		createEReference(messageSendEClass, MESSAGE_SEND__EXPRESSION_LISTS);

		modelEClass = createEClass(MODEL);
		createEAttribute(modelEClass, MODEL__NAME);
		createEReference(modelEClass, MODEL__CLASSES);
		createEReference(modelEClass, MODEL__PROCESS_INSTANCES);
		createEReference(modelEClass, MODEL__CHANNELS);
		createEReference(modelEClass, MODEL__PRIMITIVE_TYPES);

		outgoingMessageEClass = createEClass(OUTGOING_MESSAGE);

		parallelCompositionEClass = createEClass(PARALLEL_COMPOSITION);
		createEReference(parallelCompositionEClass, PARALLEL_COMPOSITION__STATEMENT_LISTS);

		parameterEClass = createEClass(PARAMETER);
		createEReference(parameterEClass, PARAMETER__TYPE);
		createEAttribute(parameterEClass, PARAMETER__NAME);

		parameterExpressionEClass = createEClass(PARAMETER_EXPRESSION);
		createEReference(parameterExpressionEClass, PARAMETER_EXPRESSION__PARAMETER);

		primitiveTypeEClass = createEClass(PRIMITIVE_TYPE);
		createEAttribute(primitiveTypeEClass, PRIMITIVE_TYPE__PRIMITIVE_TYPE_ENUM);

		processClassEClass = createEClass(PROCESS_CLASS);
		createEReference(processClassEClass, PROCESS_CLASS__VARIABLES);
		createEReference(processClassEClass, PROCESS_CLASS__METHODS);
		createEReference(processClassEClass, PROCESS_CLASS__SUPER_CLASS);
		createEReference(processClassEClass, PROCESS_CLASS__PARAMETERS);
		createEReference(processClassEClass, PROCESS_CLASS__PORTS);
		createEReference(processClassEClass, PROCESS_CLASS__INITIAL_METHOD_CALL);

		processInstanceEClass = createEClass(PROCESS_INSTANCE);
		createEAttribute(processInstanceEClass, PROCESS_INSTANCE__NAME);
		createEReference(processInstanceEClass, PROCESS_INSTANCE__CLASS);

		processMethodEClass = createEClass(PROCESS_METHOD);
		createEReference(processMethodEClass, PROCESS_METHOD__INPUT_PARAMETERS);
		createEReference(processMethodEClass, PROCESS_METHOD__OUTPUT_PARAMETERS);
		createEReference(processMethodEClass, PROCESS_METHOD__VARIABLES);
		createEAttribute(processMethodEClass, PROCESS_METHOD__NAME);
		createEReference(processMethodEClass, PROCESS_METHOD__BODY);

		processMethodCallEClass = createEClass(PROCESS_METHOD_CALL);
		createEReference(processMethodCallEClass, PROCESS_METHOD_CALL__METHOD);
		createEReference(processMethodCallEClass, PROCESS_METHOD_CALL__PARAMETER_EXPRESSIONS);
		createEReference(processMethodCallEClass, PROCESS_METHOD_CALL__VARIABLES);

		portEClass = createEClass(PORT);
		createEAttribute(portEClass, PORT__NAME);

		referenceSelfEClass = createEClass(REFERENCE_SELF);

		returnExpressionEClass = createEClass(RETURN_EXPRESSION);
		createEReference(returnExpressionEClass, RETURN_EXPRESSION__RETURN_EXPRESSIONS);

		selectionEClass = createEClass(SELECTION);
		createEReference(selectionEClass, SELECTION__STATEMENT_LISTS);

		skipEClass = createEClass(SKIP);

		statementEClass = createEClass(STATEMENT);

		statementListEClass = createEClass(STATEMENT_LIST);
		createEReference(statementListEClass, STATEMENT_LIST__STATEMENTS);

		timeSynchronisationEClass = createEClass(TIME_SYNCHRONISATION);
		createEReference(timeSynchronisationEClass, TIME_SYNCHRONISATION__DURATION);

		typeEClass = createEClass(TYPE);

		variableEClass = createEClass(VARIABLE);
		createEAttribute(variableEClass, VARIABLE__NAME);
		createEReference(variableEClass, VARIABLE__TYPE);

		variableExpressionEClass = createEClass(VARIABLE_EXPRESSION);
		createEReference(variableExpressionEClass, VARIABLE_EXPRESSION__VARIABLE);

		// Create enums
		primitiveTypeEnumEEnum = createEEnum(PRIMITIVE_TYPE_ENUM);
		primitiveDataMethodEnumEEnum = createEEnum(PRIMITIVE_DATA_METHOD_ENUM);
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
		abortEClass.getESuperTypes().add(this.getStatement());
		assignmentEClass.getESuperTypes().add(this.getExpression());
		classEClass.getESuperTypes().add(this.getType());
		conditionalExpressionEClass.getESuperTypes().add(this.getExpression());
		conditionalStatementEClass.getESuperTypes().add(this.getStatement());
		constantExpressionEClass.getESuperTypes().add(this.getExpression());
		currentTimeEClass.getESuperTypes().add(this.getExpression());
		dataClassEClass.getESuperTypes().add(this.getClass_());
		dataMethodCallEClass.getESuperTypes().add(this.getExpression());
		primitiveDataMethodCallEClass.getESuperTypes().add(this.getExpression());
		dataObjectCreationEClass.getESuperTypes().add(this.getExpression());
		expressionEClass.getESuperTypes().add(this.getStatement());
		guardedExecutionEClass.getESuperTypes().add(this.getStatement());
		incomingMessageEClass.getESuperTypes().add(this.getMessage());
		interruptEClass.getESuperTypes().add(this.getStatement());
		loopExpressionEClass.getESuperTypes().add(this.getExpression());
		loopStatementEClass.getESuperTypes().add(this.getStatement());
		messageReceiveEClass.getESuperTypes().add(this.getStatement());
		messageSendEClass.getESuperTypes().add(this.getStatement());
		outgoingMessageEClass.getESuperTypes().add(this.getMessage());
		parallelCompositionEClass.getESuperTypes().add(this.getStatement());
		parameterExpressionEClass.getESuperTypes().add(this.getExpression());
		primitiveTypeEClass.getESuperTypes().add(this.getType());
		processClassEClass.getESuperTypes().add(this.getClass_());
		processMethodCallEClass.getESuperTypes().add(this.getStatement());
		referenceSelfEClass.getESuperTypes().add(this.getExpression());
		returnExpressionEClass.getESuperTypes().add(this.getExpression());
		selectionEClass.getESuperTypes().add(this.getStatement());
		skipEClass.getESuperTypes().add(this.getStatement());
		timeSynchronisationEClass.getESuperTypes().add(this.getStatement());
		variableExpressionEClass.getESuperTypes().add(this.getExpression());

		// Initialize classes and features; add operations and parameters
		initEClass(abortEClass, Abort.class, "Abort", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getAbort_Body(), this.getStatement(), null, "body", null, 1, -1, Abort.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getAbort_Condition(), this.getExpression(), null, "condition", null, 1, -1, Abort.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(assignmentEClass, Assignment.class, "Assignment", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getAssignment_RightHandSide(), this.getExpression(), null, "rightHandSide", null, 1, -1, Assignment.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getAssignment_LeftHandSide(), this.getVariable(), null, "leftHandSide", null, 1, 1, Assignment.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(atomicEClass, Atomic.class, "Atomic", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getAtomic_Statements(), this.getStatement(), null, "statements", null, 1, -1, Atomic.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(channelEClass, Channel.class, "Channel", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getChannel_Name(), ecorePackage.getEString(), "name", null, 1, 1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getChannel_Port1(), this.getPort(), null, "port1", null, 1, 1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getChannel_Port2(), this.getPort(), null, "port2", null, 1, 1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getChannel_ProcessInstance1(), this.getProcessInstance(), null, "processInstance1", null, 1, 1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getChannel_ProcessInstance2(), this.getProcessInstance(), null, "processInstance2", null, 1, 1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(classEClass, poosl.Class.class, "Class", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getClass_Name(), ecorePackage.getEString(), "name", null, 0, 1, poosl.Class.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(conditionalExpressionEClass, ConditionalExpression.class, "ConditionalExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getConditionalExpression_Condition(), this.getExpression(), null, "condition", null, 1, -1, ConditionalExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getConditionalExpression_ThenClause(), this.getExpression(), null, "thenClause", null, 1, -1, ConditionalExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getConditionalExpression_ElseClause(), this.getExpression(), null, "elseClause", null, 0, -1, ConditionalExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(conditionalStatementEClass, ConditionalStatement.class, "ConditionalStatement", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getConditionalStatement_Condition(), this.getExpression(), null, "condition", null, 1, -1, ConditionalStatement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getConditionalStatement_ThenClause(), this.getStatement(), null, "thenClause", null, 1, -1, ConditionalStatement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getConditionalStatement_ElseClause(), this.getStatement(), null, "elseClause", null, 0, -1, ConditionalStatement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(constantEClass, Constant.class, "Constant", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getConstant_Integer(), ecorePackage.getEInt(), "integer", null, 0, 1, Constant.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getConstant_String(), ecorePackage.getEString(), "string", null, 0, 1, Constant.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getConstant_Boolean(), ecorePackage.getEBoolean(), "boolean", null, 0, 1, Constant.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getConstant_Real(), ecorePackage.getEFloat(), "real", null, 0, 1, Constant.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getConstant_Type(), this.getPrimitiveTypeEnum(), "type", null, 0, 1, Constant.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(constantExpressionEClass, ConstantExpression.class, "ConstantExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getConstantExpression_Constant(), this.getConstant(), null, "constant", null, 1, 1, ConstantExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(currentTimeEClass, CurrentTime.class, "CurrentTime", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(dataClassEClass, DataClass.class, "DataClass", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getDataClass_Variables(), this.getVariable(), null, "variables", null, 0, -1, DataClass.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getDataClass_Methods(), this.getDataMethod(), null, "methods", null, 0, -1, DataClass.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getDataClass_SuperClass(), this.getDataClass(), null, "superClass", null, 0, 1, DataClass.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(dataMethodEClass, DataMethod.class, "DataMethod", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getDataMethod_Parameters(), this.getParameter(), null, "parameters", null, 0, -1, DataMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getDataMethod_Name(), ecorePackage.getEString(), "name", null, 0, 1, DataMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getDataMethod_Variables(), this.getVariable(), null, "variables", null, 0, -1, DataMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getDataMethod_Body(), this.getExpression(), null, "body", null, 0, -1, DataMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getDataMethod_ReturnType(), this.getType(), null, "returnType", null, 0, 1, DataMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(dataMethodCallEClass, DataMethodCall.class, "DataMethodCall", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getDataMethodCall_Super(), ecorePackage.getEBoolean(), "super", null, 0, 1, DataMethodCall.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getDataMethodCall_ObjectExpressions(), this.getExpression(), null, "objectExpressions", null, 1, -1, DataMethodCall.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getDataMethodCall_ParameterExpressionLists(), this.getExpressionList(), null, "parameterExpressionLists", null, 0, -1, DataMethodCall.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getDataMethodCall_DataMethod(), this.getDataMethod(), null, "dataMethod", null, 1, 1, DataMethodCall.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(primitiveDataMethodCallEClass, PrimitiveDataMethodCall.class, "PrimitiveDataMethodCall", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getPrimitiveDataMethodCall_ObjectExpressions(), this.getExpression(), null, "objectExpressions", null, 1, -1, PrimitiveDataMethodCall.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getPrimitiveDataMethodCall_ParameterExpressionLists(), this.getExpressionList(), null, "parameterExpressionLists", null, 0, -1, PrimitiveDataMethodCall.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getPrimitiveDataMethodCall_PrimitiveDataMethod(), this.getPrimitiveDataMethodEnum(), "primitiveDataMethod", null, 1, 1, PrimitiveDataMethodCall.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(dataObjectCreationEClass, DataObjectCreation.class, "DataObjectCreation", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getDataObjectCreation_DataClass(), this.getDataClass(), null, "dataClass", null, 1, 1, DataObjectCreation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(expressionEClass, Expression.class, "Expression", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(expressionListEClass, ExpressionList.class, "ExpressionList", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getExpressionList_Expressions(), this.getExpression(), null, "expressions", null, 1, -1, ExpressionList.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(guardedExecutionEClass, GuardedExecution.class, "GuardedExecution", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getGuardedExecution_Statement(), this.getStatement(), null, "statement", null, 1, 1, GuardedExecution.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getGuardedExecution_Guard(), this.getExpression(), null, "guard", null, 1, -1, GuardedExecution.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(incomingMessageEClass, IncomingMessage.class, "IncomingMessage", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(interruptEClass, Interrupt.class, "Interrupt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getInterrupt_Body(), this.getStatement(), null, "body", null, 1, -1, Interrupt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getInterrupt_Condition(), this.getExpression(), null, "condition", null, 1, -1, Interrupt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(loopExpressionEClass, LoopExpression.class, "LoopExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getLoopExpression_Condition(), this.getExpression(), null, "condition", null, 1, -1, LoopExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getLoopExpression_Body(), this.getExpression(), null, "body", null, 1, -1, LoopExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(loopStatementEClass, LoopStatement.class, "LoopStatement", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getLoopStatement_Body(), this.getStatement(), null, "body", null, 1, -1, LoopStatement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getLoopStatement_Condition(), this.getExpression(), null, "condition", null, 1, -1, LoopStatement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(messageEClass, Message.class, "Message", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getMessage_Name(), ecorePackage.getEString(), "name", null, 0, 1, Message.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMessage_ParameterTypes(), this.getType(), null, "parameterTypes", null, 0, -1, Message.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(messageReceiveEClass, MessageReceive.class, "MessageReceive", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getMessageReceive_Message(), this.getIncomingMessage(), null, "message", null, 1, 1, MessageReceive.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMessageReceive_Port(), this.getPort(), null, "port", null, 1, 1, MessageReceive.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMessageReceive_Variables(), this.getVariable(), null, "variables", null, 0, -1, MessageReceive.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMessageReceive_Condition(), this.getExpression(), null, "condition", null, 0, -1, MessageReceive.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMessageReceive_Expressions(), this.getExpression(), null, "expressions", null, 0, -1, MessageReceive.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(messageSendEClass, MessageSend.class, "MessageSend", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getMessageSend_Message(), this.getOutgoingMessage(), null, "message", null, 1, 1, MessageSend.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMessageSend_Port(), this.getPort(), null, "port", null, 1, 1, MessageSend.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMessageSend_Expressions(), this.getExpression(), null, "expressions", null, 0, -1, MessageSend.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMessageSend_ExpressionLists(), this.getExpressionList(), null, "expressionLists", null, 0, -1, MessageSend.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(modelEClass, Model.class, "Model", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getModel_Name(), ecorePackage.getEString(), "name", null, 0, 1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getModel_Classes(), this.getClass_(), null, "classes", null, 0, -1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getModel_ProcessInstances(), this.getProcessInstance(), null, "processInstances", null, 0, -1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getModel_Channels(), this.getChannel(), null, "channels", null, 0, -1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getModel_PrimitiveTypes(), this.getPrimitiveType(), null, "primitiveTypes", null, 0, -1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(outgoingMessageEClass, OutgoingMessage.class, "OutgoingMessage", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(parallelCompositionEClass, ParallelComposition.class, "ParallelComposition", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getParallelComposition_StatementLists(), this.getStatementList(), null, "statementLists", null, 2, -1, ParallelComposition.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(parameterEClass, Parameter.class, "Parameter", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getParameter_Type(), this.getType(), null, "type", null, 1, 1, Parameter.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getParameter_Name(), ecorePackage.getEString(), "name", null, 0, 1, Parameter.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(parameterExpressionEClass, ParameterExpression.class, "ParameterExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getParameterExpression_Parameter(), this.getParameter(), null, "Parameter", null, 1, 1, ParameterExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(primitiveTypeEClass, PrimitiveType.class, "PrimitiveType", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getPrimitiveType_PrimitiveTypeEnum(), this.getPrimitiveTypeEnum(), "primitiveTypeEnum", null, 0, 1, PrimitiveType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(processClassEClass, ProcessClass.class, "ProcessClass", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getProcessClass_Variables(), this.getVariable(), null, "variables", null, 0, -1, ProcessClass.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessClass_Methods(), this.getProcessMethod(), null, "methods", null, 0, -1, ProcessClass.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessClass_SuperClass(), this.getProcessClass(), null, "superClass", null, 0, 1, ProcessClass.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessClass_Parameters(), this.getParameter(), null, "parameters", null, 0, -1, ProcessClass.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessClass_Ports(), this.getPort(), null, "ports", null, 0, -1, ProcessClass.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessClass_InitialMethodCall(), this.getProcessMethodCall(), null, "initialMethodCall", null, 1, 1, ProcessClass.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(processInstanceEClass, ProcessInstance.class, "ProcessInstance", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getProcessInstance_Name(), ecorePackage.getEString(), "name", null, 0, 1, ProcessInstance.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessInstance_Class(), this.getProcessClass(), null, "class", null, 1, 1, ProcessInstance.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(processMethodEClass, ProcessMethod.class, "ProcessMethod", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getProcessMethod_InputParameters(), this.getParameter(), null, "inputParameters", null, 0, -1, ProcessMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessMethod_OutputParameters(), this.getParameter(), null, "outputParameters", null, 0, -1, ProcessMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessMethod_Variables(), this.getVariable(), null, "variables", null, 0, -1, ProcessMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getProcessMethod_Name(), ecorePackage.getEString(), "name", null, 0, 1, ProcessMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessMethod_Body(), this.getStatement(), null, "body", null, 1, -1, ProcessMethod.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(processMethodCallEClass, ProcessMethodCall.class, "ProcessMethodCall", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getProcessMethodCall_Method(), this.getProcessMethod(), null, "method", null, 1, 1, ProcessMethodCall.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessMethodCall_ParameterExpressions(), this.getExpressionList(), null, "parameterExpressions", null, 0, -1, ProcessMethodCall.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProcessMethodCall_Variables(), this.getVariable(), null, "variables", null, 0, -1, ProcessMethodCall.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(portEClass, Port.class, "Port", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getPort_Name(), ecorePackage.getEString(), "name", null, 0, 1, Port.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(referenceSelfEClass, ReferenceSelf.class, "ReferenceSelf", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(returnExpressionEClass, ReturnExpression.class, "ReturnExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getReturnExpression_ReturnExpressions(), this.getExpression(), null, "returnExpressions", null, 1, -1, ReturnExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(selectionEClass, Selection.class, "Selection", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getSelection_StatementLists(), this.getStatementList(), null, "statementLists", null, 2, -1, Selection.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(skipEClass, Skip.class, "Skip", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(statementEClass, Statement.class, "Statement", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(statementListEClass, StatementList.class, "StatementList", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getStatementList_Statements(), this.getStatement(), null, "statements", null, 1, -1, StatementList.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(timeSynchronisationEClass, TimeSynchronisation.class, "TimeSynchronisation", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getTimeSynchronisation_Duration(), this.getExpression(), null, "duration", null, 1, -1, TimeSynchronisation.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(typeEClass, Type.class, "Type", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(variableEClass, Variable.class, "Variable", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getVariable_Name(), ecorePackage.getEString(), "name", null, 0, 1, Variable.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getVariable_Type(), this.getType(), null, "type", null, 1, 1, Variable.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(variableExpressionEClass, VariableExpression.class, "VariableExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getVariableExpression_Variable(), this.getVariable(), null, "variable", null, 1, 1, VariableExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		// Initialize enums and add enum literals
		initEEnum(primitiveTypeEnumEEnum, PrimitiveTypeEnum.class, "PrimitiveTypeEnum");
		addEEnumLiteral(primitiveTypeEnumEEnum, PrimitiveTypeEnum.STRING);
		addEEnumLiteral(primitiveTypeEnumEEnum, PrimitiveTypeEnum.BOOLEAN);
		addEEnumLiteral(primitiveTypeEnumEEnum, PrimitiveTypeEnum.INTEGER);
		addEEnumLiteral(primitiveTypeEnumEEnum, PrimitiveTypeEnum.REAL);

		initEEnum(primitiveDataMethodEnumEEnum, PrimitiveDataMethodEnum.class, "PrimitiveDataMethodEnum");
		addEEnumLiteral(primitiveDataMethodEnumEEnum, PrimitiveDataMethodEnum.ADD);
		addEEnumLiteral(primitiveDataMethodEnumEEnum, PrimitiveDataMethodEnum.AT_LEAST);
		addEEnumLiteral(primitiveDataMethodEnumEEnum, PrimitiveDataMethodEnum.AND);
		addEEnumLiteral(primitiveDataMethodEnumEEnum, PrimitiveDataMethodEnum.EQUALS);
		addEEnumLiteral(primitiveDataMethodEnumEEnum, PrimitiveDataMethodEnum.DIFFERS);
		addEEnumLiteral(primitiveDataMethodEnumEEnum, PrimitiveDataMethodEnum.SUBTRACT);
		addEEnumLiteral(primitiveDataMethodEnumEEnum, PrimitiveDataMethodEnum.AT_MOST);
		addEEnumLiteral(primitiveDataMethodEnumEEnum, PrimitiveDataMethodEnum.OR);

		// Create resource
		createResource(eNS_URI);
	}

} //PooslPackageImpl
