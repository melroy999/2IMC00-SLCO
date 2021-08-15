/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import java.io.IOException;

import java.net.URL;

import nqc.NqcFactory;
import nqc.NqcPackage;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.common.util.WrappedException;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EClassifier;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

import org.eclipse.emf.ecore.impl.EPackageImpl;

import org.eclipse.emf.ecore.resource.Resource;

import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Package</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class NqcPackageImpl extends EPackageImpl implements NqcPackage {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected String packageFilename = "nqc.ecore";

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass acquireConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass acquireStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass arrayExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass assignmentStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass binaryExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass binaryBuiltInFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass binaryBuiltInValueFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass blockStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass booleanConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass breakStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass builtInFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass builtInValueFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass callStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass caseEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass compoundExpressionEClass = null;

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
	private EClass continueStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass controlStructureEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass directionConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass displayModeConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass doWhileStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass emptyStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass eventTypeConstantEClass = null;

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
	private EClass functionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass functionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass forStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass goToStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass ifStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass integerConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass labelEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass monitorHandlerEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass monitorStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass nullaryBuiltInFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass nullaryBuiltInValueFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass outputModeConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass outputPortNameConstantEClass = null;

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
	private EClass programEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass programsEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass quaternaryBuiltInFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass repeatStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass returnStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass senaryBuiltInFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass sensorConfigConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass sensorModeConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass sensorNameConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass sensorTypeConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass serialBaudConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass serialBiphaseConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass serialChecksumConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass serialChannelConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass serialCommConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass serialPacketConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass soundConstantEClass = null;

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
	private EClass startStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass stopStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass subroutineEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass subroutineCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass switchStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass taskEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass ternaryBuiltInFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass ternaryExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass txPowerConstantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass unaryBuiltInFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass unaryBuiltInValueFunctionCallEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass unaryExpressionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass untilStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass valueExpressionEClass = null;

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
	private EClass whileStatementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum acquireEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum assignmentStatementEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum binaryOperatorEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum builtInBinaryFunctionEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum builtInBinaryValueFunctionEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum builtInNullaryFunctionEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum builtInNullaryValueFunctionEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum builtInUnaryValueFunctionEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum builtInUnaryFunctionEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum builtInQuaternaryFunctionEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum builtInSenaryFunctionEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum builtInTernaryFunctionEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum directionEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum displayModeEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum eventTypeEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum outputModeEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum outputPortNameEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum serialBaudEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum serialBiphaseEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum serialChannelEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum serialChecksumEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum serialCommEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum serialPacketEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum sensorConfigEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum sensorModeEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum sensorNameEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum sensorTypeEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum soundEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum ternaryOperatorEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum txPowerEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum typeEnumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum unaryOperatorEnumEEnum = null;

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
	 * @see nqc.NqcPackage#eNS_URI
	 * @see #init()
	 * @generated
	 */
	private NqcPackageImpl() {
		super(eNS_URI, NqcFactory.eINSTANCE);
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
	 * <p>This method is used to initialize {@link NqcPackage#eINSTANCE} when that field is accessed.
	 * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #eNS_URI
	 * @generated
	 */
	public static NqcPackage init() {
		if (isInited) return (NqcPackage)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI);

		// Obtain or create and register package
		NqcPackageImpl theNqcPackage = (NqcPackageImpl)(EPackage.Registry.INSTANCE.get(eNS_URI) instanceof NqcPackageImpl ? EPackage.Registry.INSTANCE.get(eNS_URI) : new NqcPackageImpl());

		isInited = true;

		// Load packages
		theNqcPackage.loadPackage();

		// Fix loaded packages
		theNqcPackage.fixPackageContents();

		// Mark meta-data to indicate it can't be changed
		theNqcPackage.freeze();

  
		// Update the registry and return the package
		EPackage.Registry.INSTANCE.put(NqcPackage.eNS_URI, theNqcPackage);
		return theNqcPackage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getAcquireConstant() {
		if (acquireConstantEClass == null) {
			acquireConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(32);
		}
		return acquireConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getAcquireConstant_Acquire() {
        return (EAttribute)getAcquireConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getAcquireStatement() {
		if (acquireStatementEClass == null) {
			acquireStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(33);
		}
		return acquireStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAcquireStatement_Resource() {
        return (EReference)getAcquireStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAcquireStatement_Body() {
        return (EReference)getAcquireStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAcquireStatement_Handler() {
        return (EReference)getAcquireStatement().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getArrayExpression() {
		if (arrayExpressionEClass == null) {
			arrayExpressionEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(34);
		}
		return arrayExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getArrayExpression_Index() {
        return (EReference)getArrayExpression().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getAssignmentStatement() {
		if (assignmentStatementEClass == null) {
			assignmentStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(35);
		}
		return assignmentStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAssignmentStatement_Variable() {
        return (EReference)getAssignmentStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAssignmentStatement_Expression() {
        return (EReference)getAssignmentStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getAssignmentStatement_Operator() {
        return (EAttribute)getAssignmentStatement().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBinaryExpression() {
		if (binaryExpressionEClass == null) {
			binaryExpressionEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(36);
		}
		return binaryExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getBinaryExpression_Operator() {
        return (EAttribute)getBinaryExpression().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBinaryExpression_Operand1() {
        return (EReference)getBinaryExpression().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBinaryExpression_Operand2() {
        return (EReference)getBinaryExpression().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBinaryBuiltInFunctionCall() {
		if (binaryBuiltInFunctionCallEClass == null) {
			binaryBuiltInFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(37);
		}
		return binaryBuiltInFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getBinaryBuiltInFunctionCall_BinaryBuiltInFunction() {
        return (EAttribute)getBinaryBuiltInFunctionCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBinaryBuiltInFunctionCall_Parameter1() {
        return (EReference)getBinaryBuiltInFunctionCall().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBinaryBuiltInFunctionCall_Parameter2() {
        return (EReference)getBinaryBuiltInFunctionCall().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBinaryBuiltInValueFunctionCall() {
		if (binaryBuiltInValueFunctionCallEClass == null) {
			binaryBuiltInValueFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(38);
		}
		return binaryBuiltInValueFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getBinaryBuiltInValueFunctionCall_BinaryBuiltInValueFunction() {
        return (EAttribute)getBinaryBuiltInValueFunctionCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBinaryBuiltInValueFunctionCall_Parameter1() {
        return (EReference)getBinaryBuiltInValueFunctionCall().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBinaryBuiltInValueFunctionCall_Parameter2() {
        return (EReference)getBinaryBuiltInValueFunctionCall().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBlockStatement() {
		if (blockStatementEClass == null) {
			blockStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(39);
		}
		return blockStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBlockStatement_LocalVariables() {
        return (EReference)getBlockStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getBlockStatement_Statements() {
        return (EReference)getBlockStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBooleanConstant() {
		if (booleanConstantEClass == null) {
			booleanConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(40);
		}
		return booleanConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getBooleanConstant_Value() {
        return (EAttribute)getBooleanConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBreakStatement() {
		if (breakStatementEClass == null) {
			breakStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(41);
		}
		return breakStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBuiltInFunctionCall() {
		if (builtInFunctionCallEClass == null) {
			builtInFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(42);
		}
		return builtInFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getBuiltInValueFunctionCall() {
		if (builtInValueFunctionCallEClass == null) {
			builtInValueFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(43);
		}
		return builtInValueFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCallStatement() {
		if (callStatementEClass == null) {
			callStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(44);
		}
		return callStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCase() {
		if (caseEClass == null) {
			caseEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(45);
		}
		return caseEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getCase_IsDefault() {
        return (EAttribute)getCase().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getCase_Label() {
        return (EReference)getCase().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getCase_Statements() {
        return (EReference)getCase().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCompoundExpression() {
		if (compoundExpressionEClass == null) {
			compoundExpressionEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(46);
		}
		return compoundExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getConstantExpression() {
		if (constantExpressionEClass == null) {
			constantExpressionEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(47);
		}
		return constantExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getContinueStatement() {
		if (continueStatementEClass == null) {
			continueStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(48);
		}
		return continueStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getControlStructure() {
		if (controlStructureEClass == null) {
			controlStructureEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(49);
		}
		return controlStructureEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getDirectionConstant() {
		if (directionConstantEClass == null) {
			directionConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(50);
		}
		return directionConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getDirectionConstant_Direction() {
        return (EAttribute)getDirectionConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getDisplayModeConstant() {
		if (displayModeConstantEClass == null) {
			displayModeConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(51);
		}
		return displayModeConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getDisplayModeConstant_DisplayMode() {
        return (EAttribute)getDisplayModeConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getDoWhileStatement() {
		if (doWhileStatementEClass == null) {
			doWhileStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(52);
		}
		return doWhileStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDoWhileStatement_Body() {
        return (EReference)getDoWhileStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getDoWhileStatement_Condition() {
        return (EReference)getDoWhileStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getEmptyStatement() {
		if (emptyStatementEClass == null) {
			emptyStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(53);
		}
		return emptyStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getEventTypeConstant() {
		if (eventTypeConstantEClass == null) {
			eventTypeConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(54);
		}
		return eventTypeConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getEventTypeConstant_EventType() {
        return (EAttribute)getEventTypeConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getExpression() {
		if (expressionEClass == null) {
			expressionEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(55);
		}
		return expressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getFunction() {
		if (functionEClass == null) {
			functionEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(56);
		}
		return functionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getFunction_Name() {
        return (EAttribute)getFunction().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getFunction_Statements() {
        return (EReference)getFunction().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getFunction_LocalVariables() {
        return (EReference)getFunction().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getFunction_Parameters() {
        return (EReference)getFunction().getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getFunctionCall() {
		if (functionCallEClass == null) {
			functionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(57);
		}
		return functionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getFunctionCall_Callee() {
        return (EReference)getFunctionCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getFunctionCall_Parameters() {
        return (EReference)getFunctionCall().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getForStatement() {
		if (forStatementEClass == null) {
			forStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(58);
		}
		return forStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getForStatement_Initialization() {
        return (EReference)getForStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getForStatement_Condition() {
        return (EReference)getForStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getForStatement_Update() {
        return (EReference)getForStatement().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getGoToStatement() {
		if (goToStatementEClass == null) {
			goToStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(59);
		}
		return goToStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getGoToStatement_JumpLabel() {
        return (EReference)getGoToStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getIfStatement() {
		if (ifStatementEClass == null) {
			ifStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(60);
		}
		return ifStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getIfStatement_Condition() {
        return (EReference)getIfStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getIfStatement_Consequence() {
        return (EReference)getIfStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getIfStatement_Alternative() {
        return (EReference)getIfStatement().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getIntegerConstant() {
		if (integerConstantEClass == null) {
			integerConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(61);
		}
		return integerConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getIntegerConstant_Value() {
        return (EAttribute)getIntegerConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getLabel() {
		if (labelEClass == null) {
			labelEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(62);
		}
		return labelEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getLabel_Label() {
        return (EAttribute)getLabel().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getMonitorHandler() {
		if (monitorHandlerEClass == null) {
			monitorHandlerEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(63);
		}
		return monitorHandlerEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMonitorHandler_CatchEvent() {
        return (EReference)getMonitorHandler().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMonitorHandler_Handler() {
        return (EReference)getMonitorHandler().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getMonitorStatement() {
		if (monitorStatementEClass == null) {
			monitorStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(64);
		}
		return monitorStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMonitorStatement_Events() {
        return (EReference)getMonitorStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMonitorStatement_Body() {
        return (EReference)getMonitorStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMonitorStatement_Handlers() {
        return (EReference)getMonitorStatement().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getNullaryBuiltInFunctionCall() {
		if (nullaryBuiltInFunctionCallEClass == null) {
			nullaryBuiltInFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(65);
		}
		return nullaryBuiltInFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getNullaryBuiltInFunctionCall_NullaryBuiltInFunction() {
        return (EAttribute)getNullaryBuiltInFunctionCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getNullaryBuiltInValueFunctionCall() {
		if (nullaryBuiltInValueFunctionCallEClass == null) {
			nullaryBuiltInValueFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(66);
		}
		return nullaryBuiltInValueFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getNullaryBuiltInValueFunctionCall_NullaryBuiltInValueFunction() {
        return (EAttribute)getNullaryBuiltInValueFunctionCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getOutputModeConstant() {
		if (outputModeConstantEClass == null) {
			outputModeConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(67);
		}
		return outputModeConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getOutputModeConstant_OutputMode() {
        return (EAttribute)getOutputModeConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getOutputPortNameConstant() {
		if (outputPortNameConstantEClass == null) {
			outputPortNameConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(68);
		}
		return outputPortNameConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getOutputPortNameConstant_OutputPortName() {
        return (EAttribute)getOutputPortNameConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getParameter() {
		if (parameterEClass == null) {
			parameterEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(69);
		}
		return parameterEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getParameter_Name() {
        return (EAttribute)getParameter().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getParameter_Type() {
        return (EAttribute)getParameter().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getProgram() {
		if (programEClass == null) {
			programEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(70);
		}
		return programEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getProgram_Name() {
        return (EAttribute)getProgram().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProgram_Tasks() {
        return (EReference)getProgram().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProgram_Functions() {
        return (EReference)getProgram().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProgram_Subroutines() {
        return (EReference)getProgram().getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProgram_GlobalVariables() {
        return (EReference)getProgram().getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getPrograms() {
		if (programsEClass == null) {
			programsEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(71);
		}
		return programsEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getPrograms_Programs() {
        return (EReference)getPrograms().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getQuaternaryBuiltInFunctionCall() {
		if (quaternaryBuiltInFunctionCallEClass == null) {
			quaternaryBuiltInFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(72);
		}
		return quaternaryBuiltInFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getQuaternaryBuiltInFunctionCall_QuaternaryBuiltInFunction() {
        return (EAttribute)getQuaternaryBuiltInFunctionCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getQuaternaryBuiltInFunctionCall_Parameter1() {
        return (EReference)getQuaternaryBuiltInFunctionCall().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getQuaternaryBuiltInFunctionCall_Parameter2() {
        return (EReference)getQuaternaryBuiltInFunctionCall().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getQuaternaryBuiltInFunctionCall_Parameter3() {
        return (EReference)getQuaternaryBuiltInFunctionCall().getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getQuaternaryBuiltInFunctionCall_Parameter4() {
        return (EReference)getQuaternaryBuiltInFunctionCall().getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getRepeatStatement() {
		if (repeatStatementEClass == null) {
			repeatStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(73);
		}
		return repeatStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getRepeatStatement_Condition() {
        return (EReference)getRepeatStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getRepeatStatement_Body() {
        return (EReference)getRepeatStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getReturnStatement() {
		if (returnStatementEClass == null) {
			returnStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(74);
		}
		return returnStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSenaryBuiltInFunctionCall() {
		if (senaryBuiltInFunctionCallEClass == null) {
			senaryBuiltInFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(75);
		}
		return senaryBuiltInFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSenaryBuiltInFunctionCall_SenaryBuiltInFunction() {
        return (EAttribute)getSenaryBuiltInFunctionCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSenaryBuiltInFunctionCall_Parameter1() {
        return (EReference)getSenaryBuiltInFunctionCall().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSenaryBuiltInFunctionCall_Parameter2() {
        return (EReference)getSenaryBuiltInFunctionCall().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSenaryBuiltInFunctionCall_Parameter3() {
        return (EReference)getSenaryBuiltInFunctionCall().getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSenaryBuiltInFunctionCall_Parameter4() {
        return (EReference)getSenaryBuiltInFunctionCall().getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSenaryBuiltInFunctionCall_Parameter5() {
        return (EReference)getSenaryBuiltInFunctionCall().getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSenaryBuiltInFunctionCall_Parameter6() {
        return (EReference)getSenaryBuiltInFunctionCall().getEStructuralFeatures().get(6);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSensorConfigConstant() {
		if (sensorConfigConstantEClass == null) {
			sensorConfigConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(76);
		}
		return sensorConfigConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSensorConfigConstant_SensorConfig() {
        return (EAttribute)getSensorConfigConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSensorModeConstant() {
		if (sensorModeConstantEClass == null) {
			sensorModeConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(77);
		}
		return sensorModeConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSensorModeConstant_SensorMode() {
        return (EAttribute)getSensorModeConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSensorNameConstant() {
		if (sensorNameConstantEClass == null) {
			sensorNameConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(78);
		}
		return sensorNameConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSensorNameConstant_SensorName() {
        return (EAttribute)getSensorNameConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSensorTypeConstant() {
		if (sensorTypeConstantEClass == null) {
			sensorTypeConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(79);
		}
		return sensorTypeConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSensorTypeConstant_SensorType() {
        return (EAttribute)getSensorTypeConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSerialBaudConstant() {
		if (serialBaudConstantEClass == null) {
			serialBaudConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(80);
		}
		return serialBaudConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSerialBaudConstant_SerialBaud() {
        return (EAttribute)getSerialBaudConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSerialBiphaseConstant() {
		if (serialBiphaseConstantEClass == null) {
			serialBiphaseConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(81);
		}
		return serialBiphaseConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSerialBiphaseConstant_SerialBiphase() {
        return (EAttribute)getSerialBiphaseConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSerialChecksumConstant() {
		if (serialChecksumConstantEClass == null) {
			serialChecksumConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(82);
		}
		return serialChecksumConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSerialChecksumConstant_SerialChecksum() {
        return (EAttribute)getSerialChecksumConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSerialChannelConstant() {
		if (serialChannelConstantEClass == null) {
			serialChannelConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(83);
		}
		return serialChannelConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSerialChannelConstant_SerialChannel() {
        return (EAttribute)getSerialChannelConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSerialCommConstant() {
		if (serialCommConstantEClass == null) {
			serialCommConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(84);
		}
		return serialCommConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSerialCommConstant_SerialComm() {
        return (EAttribute)getSerialCommConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSerialPacketConstant() {
		if (serialPacketConstantEClass == null) {
			serialPacketConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(85);
		}
		return serialPacketConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSerialPacketConstant_SerialPacket() {
        return (EAttribute)getSerialPacketConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSoundConstant() {
		if (soundConstantEClass == null) {
			soundConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(86);
		}
		return soundConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSoundConstant_Sound() {
        return (EAttribute)getSoundConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getStatement() {
		if (statementEClass == null) {
			statementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(87);
		}
		return statementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getStatement_Label() {
        return (EReference)getStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getStartStatement() {
		if (startStatementEClass == null) {
			startStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(88);
		}
		return startStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getStartStatement_Task() {
        return (EReference)getStartStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getStopStatement() {
		if (stopStatementEClass == null) {
			stopStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(89);
		}
		return stopStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getStopStatement_Task() {
        return (EReference)getStopStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSubroutine() {
		if (subroutineEClass == null) {
			subroutineEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(90);
		}
		return subroutineEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSubroutine_Name() {
        return (EAttribute)getSubroutine().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSubroutine_Statements() {
        return (EReference)getSubroutine().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSubroutine_LocalVariables() {
        return (EReference)getSubroutine().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSubroutineCall() {
		if (subroutineCallEClass == null) {
			subroutineCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(91);
		}
		return subroutineCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSubroutineCall_Callee() {
        return (EReference)getSubroutineCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSwitchStatement() {
		if (switchStatementEClass == null) {
			switchStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(92);
		}
		return switchStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSwitchStatement_Condition() {
        return (EReference)getSwitchStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSwitchStatement_Cases() {
        return (EReference)getSwitchStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getTask() {
		if (taskEClass == null) {
			taskEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(93);
		}
		return taskEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getTask_Name() {
        return (EAttribute)getTask().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTask_Statements() {
        return (EReference)getTask().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTask_LocalVariables() {
        return (EReference)getTask().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getTernaryBuiltInFunctionCall() {
		if (ternaryBuiltInFunctionCallEClass == null) {
			ternaryBuiltInFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(94);
		}
		return ternaryBuiltInFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getTernaryBuiltInFunctionCall_TernaryBuiltInFunction() {
        return (EAttribute)getTernaryBuiltInFunctionCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTernaryBuiltInFunctionCall_Parameter1() {
        return (EReference)getTernaryBuiltInFunctionCall().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTernaryBuiltInFunctionCall_Parameter2() {
        return (EReference)getTernaryBuiltInFunctionCall().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTernaryBuiltInFunctionCall_Parameter3() {
        return (EReference)getTernaryBuiltInFunctionCall().getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getTernaryExpression() {
		if (ternaryExpressionEClass == null) {
			ternaryExpressionEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(95);
		}
		return ternaryExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getTernaryExpression_Operator() {
        return (EAttribute)getTernaryExpression().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTernaryExpression_Operand1() {
        return (EReference)getTernaryExpression().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTernaryExpression_Operand2() {
        return (EReference)getTernaryExpression().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTernaryExpression_Operand3() {
        return (EReference)getTernaryExpression().getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getTxPowerConstant() {
		if (txPowerConstantEClass == null) {
			txPowerConstantEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(96);
		}
		return txPowerConstantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getTxPowerConstant_TxPower() {
        return (EAttribute)getTxPowerConstant().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getUnaryBuiltInFunctionCall() {
		if (unaryBuiltInFunctionCallEClass == null) {
			unaryBuiltInFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(97);
		}
		return unaryBuiltInFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getUnaryBuiltInFunctionCall_UnaryBuiltInFunction() {
        return (EAttribute)getUnaryBuiltInFunctionCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getUnaryBuiltInFunctionCall_Parameter() {
        return (EReference)getUnaryBuiltInFunctionCall().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getUnaryBuiltInValueFunctionCall() {
		if (unaryBuiltInValueFunctionCallEClass == null) {
			unaryBuiltInValueFunctionCallEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(98);
		}
		return unaryBuiltInValueFunctionCallEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getUnaryBuiltInValueFunctionCall_UnaryBuiltInValueFunction() {
        return (EAttribute)getUnaryBuiltInValueFunctionCall().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getUnaryBuiltInValueFunctionCall_Parameter() {
        return (EReference)getUnaryBuiltInValueFunctionCall().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getUnaryExpression() {
		if (unaryExpressionEClass == null) {
			unaryExpressionEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(99);
		}
		return unaryExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getUnaryExpression_Operand() {
        return (EReference)getUnaryExpression().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getUnaryExpression_Operator() {
        return (EAttribute)getUnaryExpression().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getUntilStatement() {
		if (untilStatementEClass == null) {
			untilStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(100);
		}
		return untilStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getUntilStatement_Condition() {
        return (EReference)getUntilStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getUntilStatement_Body() {
        return (EReference)getUntilStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getValueExpression() {
		if (valueExpressionEClass == null) {
			valueExpressionEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(101);
		}
		return valueExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getVariable() {
		if (variableEClass == null) {
			variableEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(102);
		}
		return variableEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getVariable_Name() {
        return (EAttribute)getVariable().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getVariable_Type() {
        return (EAttribute)getVariable().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getVariable_InitialValue() {
        return (EReference)getVariable().getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getVariable_ArrayMaxSize() {
        return (EReference)getVariable().getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getVariableExpression() {
		if (variableExpressionEClass == null) {
			variableExpressionEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(103);
		}
		return variableExpressionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getVariableExpression_Variable() {
        return (EReference)getVariableExpression().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getWhileStatement() {
		if (whileStatementEClass == null) {
			whileStatementEClass = (EClass)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(104);
		}
		return whileStatementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getWhileStatement_Condition() {
        return (EReference)getWhileStatement().getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getWhileStatement_Body() {
        return (EReference)getWhileStatement().getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getAcquireEnum() {
		if (acquireEnumEEnum == null) {
			acquireEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(0);
		}
		return acquireEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getAssignmentStatementEnum() {
		if (assignmentStatementEnumEEnum == null) {
			assignmentStatementEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(1);
		}
		return assignmentStatementEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getBinaryOperatorEnum() {
		if (binaryOperatorEnumEEnum == null) {
			binaryOperatorEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(2);
		}
		return binaryOperatorEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getBuiltInBinaryFunctionEnum() {
		if (builtInBinaryFunctionEnumEEnum == null) {
			builtInBinaryFunctionEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(3);
		}
		return builtInBinaryFunctionEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getBuiltInBinaryValueFunctionEnum() {
		if (builtInBinaryValueFunctionEnumEEnum == null) {
			builtInBinaryValueFunctionEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(4);
		}
		return builtInBinaryValueFunctionEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getBuiltInNullaryFunctionEnum() {
		if (builtInNullaryFunctionEnumEEnum == null) {
			builtInNullaryFunctionEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(5);
		}
		return builtInNullaryFunctionEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getBuiltInNullaryValueFunctionEnum() {
		if (builtInNullaryValueFunctionEnumEEnum == null) {
			builtInNullaryValueFunctionEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(6);
		}
		return builtInNullaryValueFunctionEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getBuiltInUnaryValueFunctionEnum() {
		if (builtInUnaryValueFunctionEnumEEnum == null) {
			builtInUnaryValueFunctionEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(7);
		}
		return builtInUnaryValueFunctionEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getBuiltInUnaryFunctionEnum() {
		if (builtInUnaryFunctionEnumEEnum == null) {
			builtInUnaryFunctionEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(8);
		}
		return builtInUnaryFunctionEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getBuiltInQuaternaryFunctionEnum() {
		if (builtInQuaternaryFunctionEnumEEnum == null) {
			builtInQuaternaryFunctionEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(9);
		}
		return builtInQuaternaryFunctionEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getBuiltInSenaryFunctionEnum() {
		if (builtInSenaryFunctionEnumEEnum == null) {
			builtInSenaryFunctionEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(10);
		}
		return builtInSenaryFunctionEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getBuiltInTernaryFunctionEnum() {
		if (builtInTernaryFunctionEnumEEnum == null) {
			builtInTernaryFunctionEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(11);
		}
		return builtInTernaryFunctionEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getDirectionEnum() {
		if (directionEnumEEnum == null) {
			directionEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(12);
		}
		return directionEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getDisplayModeEnum() {
		if (displayModeEnumEEnum == null) {
			displayModeEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(13);
		}
		return displayModeEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getEventTypeEnum() {
		if (eventTypeEnumEEnum == null) {
			eventTypeEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(14);
		}
		return eventTypeEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getOutputModeEnum() {
		if (outputModeEnumEEnum == null) {
			outputModeEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(15);
		}
		return outputModeEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getOutputPortNameEnum() {
		if (outputPortNameEnumEEnum == null) {
			outputPortNameEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(16);
		}
		return outputPortNameEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSerialBaudEnum() {
		if (serialBaudEnumEEnum == null) {
			serialBaudEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(17);
		}
		return serialBaudEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSerialBiphaseEnum() {
		if (serialBiphaseEnumEEnum == null) {
			serialBiphaseEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(18);
		}
		return serialBiphaseEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSerialChannelEnum() {
		if (serialChannelEnumEEnum == null) {
			serialChannelEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(19);
		}
		return serialChannelEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSerialChecksumEnum() {
		if (serialChecksumEnumEEnum == null) {
			serialChecksumEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(20);
		}
		return serialChecksumEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSerialCommEnum() {
		if (serialCommEnumEEnum == null) {
			serialCommEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(21);
		}
		return serialCommEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSerialPacketEnum() {
		if (serialPacketEnumEEnum == null) {
			serialPacketEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(22);
		}
		return serialPacketEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSensorConfigEnum() {
		if (sensorConfigEnumEEnum == null) {
			sensorConfigEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(23);
		}
		return sensorConfigEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSensorModeEnum() {
		if (sensorModeEnumEEnum == null) {
			sensorModeEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(24);
		}
		return sensorModeEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSensorNameEnum() {
		if (sensorNameEnumEEnum == null) {
			sensorNameEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(25);
		}
		return sensorNameEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSensorTypeEnum() {
		if (sensorTypeEnumEEnum == null) {
			sensorTypeEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(26);
		}
		return sensorTypeEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getSoundEnum() {
		if (soundEnumEEnum == null) {
			soundEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(27);
		}
		return soundEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getTernaryOperatorEnum() {
		if (ternaryOperatorEnumEEnum == null) {
			ternaryOperatorEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(28);
		}
		return ternaryOperatorEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getTxPowerEnum() {
		if (txPowerEnumEEnum == null) {
			txPowerEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(29);
		}
		return txPowerEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getTypeEnum() {
		if (typeEnumEEnum == null) {
			typeEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(30);
		}
		return typeEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getUnaryOperatorEnum() {
		if (unaryOperatorEnumEEnum == null) {
			unaryOperatorEnumEEnum = (EEnum)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI).getEClassifiers().get(31);
		}
		return unaryOperatorEnumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NqcFactory getNqcFactory() {
		return (NqcFactory)getEFactoryInstance();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private boolean isLoaded = false;

	/**
	 * Laods the package and any sub-packages from their serialized form.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void loadPackage() {
		if (isLoaded) return;
		isLoaded = true;

		URL url = getClass().getResource(packageFilename);
		if (url == null) {
			throw new RuntimeException("Missing serialized package: " + packageFilename);
		}
		URI uri = URI.createURI(url.toString());
		Resource resource = new EcoreResourceFactoryImpl().createResource(uri);
		try {
			resource.load(null);
		}
		catch (IOException exception) {
			throw new WrappedException(exception);
		}
		initializeFromLoadedEPackage(this, (EPackage)resource.getContents().get(0));
		createResource(eNS_URI);
	}


	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private boolean isFixed = false;

	/**
	 * Fixes up the loaded package, to make it appear as if it had been programmatically built.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void fixPackageContents() {
		if (isFixed) return;
		isFixed = true;
		fixEClassifiers();
	}

	/**
	 * Sets the instance class on the given classifier.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected void fixInstanceClass(EClassifier eClassifier) {
		if (eClassifier.getInstanceClassName() == null) {
			eClassifier.setInstanceClassName("nqc." + eClassifier.getName());
			setGeneratedClassName(eClassifier);
		}
	}

} //NqcPackageImpl
