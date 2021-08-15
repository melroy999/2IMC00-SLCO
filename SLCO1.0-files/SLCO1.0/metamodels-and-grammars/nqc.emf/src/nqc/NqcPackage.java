/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

/**
 * <!-- begin-user-doc -->
 * The <b>Package</b> for the model.
 * It contains accessors for the meta objects to represent
 * <ul>
 *   <li>each class,</li>
 *   <li>each feature of each class,</li>
 *   <li>each enum,</li>
 *   <li>and each data type</li>
 * </ul>
 * <!-- end-user-doc -->
 * @see nqc.NqcFactory
 * @model kind="package"
 * @generated
 */
public interface NqcPackage extends EPackage {
	/**
	 * The package name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "nqc";

	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "nqc";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_PREFIX = "nqc";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	NqcPackage eINSTANCE = nqc.impl.NqcPackageImpl.init();

	/**
	 * The meta object id for the '{@link nqc.impl.StatementImpl <em>Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.StatementImpl
	 * @see nqc.impl.NqcPackageImpl#getStatement()
	 * @generated
	 */
	int STATEMENT = 55;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATEMENT__LABEL = 0;

	/**
	 * The number of structural features of the '<em>Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATEMENT_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link nqc.impl.ExpressionImpl <em>Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ExpressionImpl
	 * @see nqc.impl.NqcPackageImpl#getExpression()
	 * @generated
	 */
	int EXPRESSION = 23;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EXPRESSION__LABEL = STATEMENT__LABEL;

	/**
	 * The number of structural features of the '<em>Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EXPRESSION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.ValueExpressionImpl <em>Value Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ValueExpressionImpl
	 * @see nqc.impl.NqcPackageImpl#getValueExpression()
	 * @generated
	 */
	int VALUE_EXPRESSION = 69;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VALUE_EXPRESSION__LABEL = EXPRESSION__LABEL;

	/**
	 * The number of structural features of the '<em>Value Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VALUE_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.ConstantExpressionImpl <em>Constant Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ConstantExpressionImpl
	 * @see nqc.impl.NqcPackageImpl#getConstantExpression()
	 * @generated
	 */
	int CONSTANT_EXPRESSION = 15;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT_EXPRESSION__LABEL = VALUE_EXPRESSION__LABEL;

	/**
	 * The number of structural features of the '<em>Constant Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT_EXPRESSION_FEATURE_COUNT = VALUE_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.AcquireConstantImpl <em>Acquire Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.AcquireConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getAcquireConstant()
	 * @generated
	 */
	int ACQUIRE_CONSTANT = 0;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ACQUIRE_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Acquire</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ACQUIRE_CONSTANT__ACQUIRE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Acquire Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ACQUIRE_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.AcquireStatementImpl <em>Acquire Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.AcquireStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getAcquireStatement()
	 * @generated
	 */
	int ACQUIRE_STATEMENT = 1;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ACQUIRE_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The feature id for the '<em><b>Resource</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ACQUIRE_STATEMENT__RESOURCE = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Body</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ACQUIRE_STATEMENT__BODY = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Handler</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ACQUIRE_STATEMENT__HANDLER = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Acquire Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ACQUIRE_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link nqc.impl.VariableExpressionImpl <em>Variable Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.VariableExpressionImpl
	 * @see nqc.impl.NqcPackageImpl#getVariableExpression()
	 * @generated
	 */
	int VARIABLE_EXPRESSION = 71;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE_EXPRESSION__LABEL = VALUE_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Variable</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE_EXPRESSION__VARIABLE = VALUE_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Variable Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE_EXPRESSION_FEATURE_COUNT = VALUE_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.ArrayExpressionImpl <em>Array Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ArrayExpressionImpl
	 * @see nqc.impl.NqcPackageImpl#getArrayExpression()
	 * @generated
	 */
	int ARRAY_EXPRESSION = 2;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARRAY_EXPRESSION__LABEL = VARIABLE_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Variable</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARRAY_EXPRESSION__VARIABLE = VARIABLE_EXPRESSION__VARIABLE;

	/**
	 * The feature id for the '<em><b>Index</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARRAY_EXPRESSION__INDEX = VARIABLE_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Array Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARRAY_EXPRESSION_FEATURE_COUNT = VARIABLE_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.AssignmentStatementImpl <em>Assignment Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.AssignmentStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getAssignmentStatement()
	 * @generated
	 */
	int ASSIGNMENT_STATEMENT = 3;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGNMENT_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The feature id for the '<em><b>Variable</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGNMENT_STATEMENT__VARIABLE = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Expression</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGNMENT_STATEMENT__EXPRESSION = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Operator</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGNMENT_STATEMENT__OPERATOR = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Assignment Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGNMENT_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link nqc.impl.CompoundExpressionImpl <em>Compound Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.CompoundExpressionImpl
	 * @see nqc.impl.NqcPackageImpl#getCompoundExpression()
	 * @generated
	 */
	int COMPOUND_EXPRESSION = 14;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPOUND_EXPRESSION__LABEL = EXPRESSION__LABEL;

	/**
	 * The number of structural features of the '<em>Compound Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPOUND_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.BinaryExpressionImpl <em>Binary Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.BinaryExpressionImpl
	 * @see nqc.impl.NqcPackageImpl#getBinaryExpression()
	 * @generated
	 */
	int BINARY_EXPRESSION = 4;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_EXPRESSION__LABEL = COMPOUND_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Operator</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_EXPRESSION__OPERATOR = COMPOUND_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Operand1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_EXPRESSION__OPERAND1 = COMPOUND_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Operand2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_EXPRESSION__OPERAND2 = COMPOUND_EXPRESSION_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Binary Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_EXPRESSION_FEATURE_COUNT = COMPOUND_EXPRESSION_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link nqc.impl.CallStatementImpl <em>Call Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.CallStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getCallStatement()
	 * @generated
	 */
	int CALL_STATEMENT = 12;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CALL_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The number of structural features of the '<em>Call Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CALL_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.BuiltInFunctionCallImpl <em>Built In Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.BuiltInFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getBuiltInFunctionCall()
	 * @generated
	 */
	int BUILT_IN_FUNCTION_CALL = 10;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BUILT_IN_FUNCTION_CALL__LABEL = CALL_STATEMENT__LABEL;

	/**
	 * The number of structural features of the '<em>Built In Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BUILT_IN_FUNCTION_CALL_FEATURE_COUNT = CALL_STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.BinaryBuiltInFunctionCallImpl <em>Binary Built In Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.BinaryBuiltInFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getBinaryBuiltInFunctionCall()
	 * @generated
	 */
	int BINARY_BUILT_IN_FUNCTION_CALL = 5;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_BUILT_IN_FUNCTION_CALL__LABEL = BUILT_IN_FUNCTION_CALL__LABEL;

	/**
	 * The feature id for the '<em><b>Binary Built In Function</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_BUILT_IN_FUNCTION_CALL__BINARY_BUILT_IN_FUNCTION = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Parameter1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_BUILT_IN_FUNCTION_CALL__PARAMETER1 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Parameter2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_BUILT_IN_FUNCTION_CALL__PARAMETER2 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Binary Built In Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_BUILT_IN_FUNCTION_CALL_FEATURE_COUNT = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link nqc.impl.BuiltInValueFunctionCallImpl <em>Built In Value Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.BuiltInValueFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getBuiltInValueFunctionCall()
	 * @generated
	 */
	int BUILT_IN_VALUE_FUNCTION_CALL = 11;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BUILT_IN_VALUE_FUNCTION_CALL__LABEL = EXPRESSION__LABEL;

	/**
	 * The number of structural features of the '<em>Built In Value Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.BinaryBuiltInValueFunctionCallImpl <em>Binary Built In Value Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.BinaryBuiltInValueFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getBinaryBuiltInValueFunctionCall()
	 * @generated
	 */
	int BINARY_BUILT_IN_VALUE_FUNCTION_CALL = 6;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_BUILT_IN_VALUE_FUNCTION_CALL__LABEL = BUILT_IN_VALUE_FUNCTION_CALL__LABEL;

	/**
	 * The feature id for the '<em><b>Binary Built In Value Function</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_BUILT_IN_VALUE_FUNCTION_CALL__BINARY_BUILT_IN_VALUE_FUNCTION = BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Parameter1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER1 = BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Parameter2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER2 = BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Binary Built In Value Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT = BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link nqc.impl.BlockStatementImpl <em>Block Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.BlockStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getBlockStatement()
	 * @generated
	 */
	int BLOCK_STATEMENT = 7;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BLOCK_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The feature id for the '<em><b>Local Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BLOCK_STATEMENT__LOCAL_VARIABLES = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Statements</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BLOCK_STATEMENT__STATEMENTS = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Block Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BLOCK_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link nqc.impl.BooleanConstantImpl <em>Boolean Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.BooleanConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getBooleanConstant()
	 * @generated
	 */
	int BOOLEAN_CONSTANT = 8;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BOOLEAN_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Value</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BOOLEAN_CONSTANT__VALUE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Boolean Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BOOLEAN_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.BreakStatementImpl <em>Break Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.BreakStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getBreakStatement()
	 * @generated
	 */
	int BREAK_STATEMENT = 9;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BREAK_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The number of structural features of the '<em>Break Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BREAK_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.CaseImpl <em>Case</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.CaseImpl
	 * @see nqc.impl.NqcPackageImpl#getCase()
	 * @generated
	 */
	int CASE = 13;

	/**
	 * The feature id for the '<em><b>Is Default</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CASE__IS_DEFAULT = 0;

	/**
	 * The feature id for the '<em><b>Label</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CASE__LABEL = 1;

	/**
	 * The feature id for the '<em><b>Statements</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CASE__STATEMENTS = 2;

	/**
	 * The number of structural features of the '<em>Case</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CASE_FEATURE_COUNT = 3;

	/**
	 * The meta object id for the '{@link nqc.impl.ContinueStatementImpl <em>Continue Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ContinueStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getContinueStatement()
	 * @generated
	 */
	int CONTINUE_STATEMENT = 16;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONTINUE_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The number of structural features of the '<em>Continue Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONTINUE_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.ControlStructureImpl <em>Control Structure</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ControlStructureImpl
	 * @see nqc.impl.NqcPackageImpl#getControlStructure()
	 * @generated
	 */
	int CONTROL_STRUCTURE = 17;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONTROL_STRUCTURE__LABEL = STATEMENT__LABEL;

	/**
	 * The number of structural features of the '<em>Control Structure</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONTROL_STRUCTURE_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.DirectionConstantImpl <em>Direction Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.DirectionConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getDirectionConstant()
	 * @generated
	 */
	int DIRECTION_CONSTANT = 18;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DIRECTION_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Direction</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DIRECTION_CONSTANT__DIRECTION = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Direction Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DIRECTION_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.DisplayModeConstantImpl <em>Display Mode Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.DisplayModeConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getDisplayModeConstant()
	 * @generated
	 */
	int DISPLAY_MODE_CONSTANT = 19;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DISPLAY_MODE_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Display Mode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DISPLAY_MODE_CONSTANT__DISPLAY_MODE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Display Mode Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DISPLAY_MODE_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.DoWhileStatementImpl <em>Do While Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.DoWhileStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getDoWhileStatement()
	 * @generated
	 */
	int DO_WHILE_STATEMENT = 20;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DO_WHILE_STATEMENT__LABEL = CONTROL_STRUCTURE__LABEL;

	/**
	 * The feature id for the '<em><b>Body</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DO_WHILE_STATEMENT__BODY = CONTROL_STRUCTURE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DO_WHILE_STATEMENT__CONDITION = CONTROL_STRUCTURE_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Do While Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DO_WHILE_STATEMENT_FEATURE_COUNT = CONTROL_STRUCTURE_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link nqc.impl.EmptyStatementImpl <em>Empty Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.EmptyStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getEmptyStatement()
	 * @generated
	 */
	int EMPTY_STATEMENT = 21;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EMPTY_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The number of structural features of the '<em>Empty Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EMPTY_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.EventTypeConstantImpl <em>Event Type Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.EventTypeConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getEventTypeConstant()
	 * @generated
	 */
	int EVENT_TYPE_CONSTANT = 22;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EVENT_TYPE_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Event Type</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EVENT_TYPE_CONSTANT__EVENT_TYPE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Event Type Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EVENT_TYPE_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.FunctionImpl <em>Function</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.FunctionImpl
	 * @see nqc.impl.NqcPackageImpl#getFunction()
	 * @generated
	 */
	int FUNCTION = 24;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION__NAME = 0;

	/**
	 * The feature id for the '<em><b>Statements</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION__STATEMENTS = 1;

	/**
	 * The feature id for the '<em><b>Local Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION__LOCAL_VARIABLES = 2;

	/**
	 * The feature id for the '<em><b>Parameters</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION__PARAMETERS = 3;

	/**
	 * The number of structural features of the '<em>Function</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION_FEATURE_COUNT = 4;

	/**
	 * The meta object id for the '{@link nqc.impl.FunctionCallImpl <em>Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.FunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getFunctionCall()
	 * @generated
	 */
	int FUNCTION_CALL = 25;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION_CALL__LABEL = CALL_STATEMENT__LABEL;

	/**
	 * The feature id for the '<em><b>Callee</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION_CALL__CALLEE = CALL_STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Parameters</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION_CALL__PARAMETERS = CALL_STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FUNCTION_CALL_FEATURE_COUNT = CALL_STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link nqc.impl.ForStatementImpl <em>For Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ForStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getForStatement()
	 * @generated
	 */
	int FOR_STATEMENT = 26;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FOR_STATEMENT__LABEL = CONTROL_STRUCTURE__LABEL;

	/**
	 * The feature id for the '<em><b>Initialization</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FOR_STATEMENT__INITIALIZATION = CONTROL_STRUCTURE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FOR_STATEMENT__CONDITION = CONTROL_STRUCTURE_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Update</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FOR_STATEMENT__UPDATE = CONTROL_STRUCTURE_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>For Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FOR_STATEMENT_FEATURE_COUNT = CONTROL_STRUCTURE_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link nqc.impl.GoToStatementImpl <em>Go To Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.GoToStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getGoToStatement()
	 * @generated
	 */
	int GO_TO_STATEMENT = 27;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int GO_TO_STATEMENT__LABEL = CONTROL_STRUCTURE__LABEL;

	/**
	 * The feature id for the '<em><b>Jump Label</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int GO_TO_STATEMENT__JUMP_LABEL = CONTROL_STRUCTURE_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Go To Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int GO_TO_STATEMENT_FEATURE_COUNT = CONTROL_STRUCTURE_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.IfStatementImpl <em>If Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.IfStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getIfStatement()
	 * @generated
	 */
	int IF_STATEMENT = 28;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IF_STATEMENT__LABEL = CONTROL_STRUCTURE__LABEL;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IF_STATEMENT__CONDITION = CONTROL_STRUCTURE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Consequence</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IF_STATEMENT__CONSEQUENCE = CONTROL_STRUCTURE_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Alternative</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IF_STATEMENT__ALTERNATIVE = CONTROL_STRUCTURE_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>If Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IF_STATEMENT_FEATURE_COUNT = CONTROL_STRUCTURE_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link nqc.impl.IntegerConstantImpl <em>Integer Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.IntegerConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getIntegerConstant()
	 * @generated
	 */
	int INTEGER_CONSTANT = 29;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTEGER_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Value</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTEGER_CONSTANT__VALUE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Integer Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTEGER_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.LabelImpl <em>Label</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.LabelImpl
	 * @see nqc.impl.NqcPackageImpl#getLabel()
	 * @generated
	 */
	int LABEL = 30;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LABEL__LABEL = 0;

	/**
	 * The number of structural features of the '<em>Label</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LABEL_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link nqc.impl.MonitorHandlerImpl <em>Monitor Handler</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.MonitorHandlerImpl
	 * @see nqc.impl.NqcPackageImpl#getMonitorHandler()
	 * @generated
	 */
	int MONITOR_HANDLER = 31;

	/**
	 * The feature id for the '<em><b>Catch Event</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITOR_HANDLER__CATCH_EVENT = 0;

	/**
	 * The feature id for the '<em><b>Handler</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITOR_HANDLER__HANDLER = 1;

	/**
	 * The number of structural features of the '<em>Monitor Handler</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITOR_HANDLER_FEATURE_COUNT = 2;

	/**
	 * The meta object id for the '{@link nqc.impl.MonitorStatementImpl <em>Monitor Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.MonitorStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getMonitorStatement()
	 * @generated
	 */
	int MONITOR_STATEMENT = 32;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITOR_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The feature id for the '<em><b>Events</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITOR_STATEMENT__EVENTS = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Body</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITOR_STATEMENT__BODY = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Handlers</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITOR_STATEMENT__HANDLERS = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Monitor Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITOR_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link nqc.impl.NullaryBuiltInFunctionCallImpl <em>Nullary Built In Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.NullaryBuiltInFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getNullaryBuiltInFunctionCall()
	 * @generated
	 */
	int NULLARY_BUILT_IN_FUNCTION_CALL = 33;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NULLARY_BUILT_IN_FUNCTION_CALL__LABEL = BUILT_IN_FUNCTION_CALL__LABEL;

	/**
	 * The feature id for the '<em><b>Nullary Built In Function</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NULLARY_BUILT_IN_FUNCTION_CALL__NULLARY_BUILT_IN_FUNCTION = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Nullary Built In Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NULLARY_BUILT_IN_FUNCTION_CALL_FEATURE_COUNT = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.NullaryBuiltInValueFunctionCallImpl <em>Nullary Built In Value Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.NullaryBuiltInValueFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getNullaryBuiltInValueFunctionCall()
	 * @generated
	 */
	int NULLARY_BUILT_IN_VALUE_FUNCTION_CALL = 34;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NULLARY_BUILT_IN_VALUE_FUNCTION_CALL__LABEL = BUILT_IN_VALUE_FUNCTION_CALL__LABEL;

	/**
	 * The feature id for the '<em><b>Nullary Built In Value Function</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NULLARY_BUILT_IN_VALUE_FUNCTION_CALL__NULLARY_BUILT_IN_VALUE_FUNCTION = BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Nullary Built In Value Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NULLARY_BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT = BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.OutputModeConstantImpl <em>Output Mode Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.OutputModeConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getOutputModeConstant()
	 * @generated
	 */
	int OUTPUT_MODE_CONSTANT = 35;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OUTPUT_MODE_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Output Mode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OUTPUT_MODE_CONSTANT__OUTPUT_MODE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Output Mode Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OUTPUT_MODE_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.OutputPortNameConstantImpl <em>Output Port Name Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.OutputPortNameConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getOutputPortNameConstant()
	 * @generated
	 */
	int OUTPUT_PORT_NAME_CONSTANT = 36;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OUTPUT_PORT_NAME_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Output Port Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OUTPUT_PORT_NAME_CONSTANT__OUTPUT_PORT_NAME = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Output Port Name Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OUTPUT_PORT_NAME_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.ParameterImpl <em>Parameter</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ParameterImpl
	 * @see nqc.impl.NqcPackageImpl#getParameter()
	 * @generated
	 */
	int PARAMETER = 37;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__NAME = 0;

	/**
	 * The feature id for the '<em><b>Type</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__TYPE = 1;

	/**
	 * The number of structural features of the '<em>Parameter</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_FEATURE_COUNT = 2;

	/**
	 * The meta object id for the '{@link nqc.impl.ProgramImpl <em>Program</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ProgramImpl
	 * @see nqc.impl.NqcPackageImpl#getProgram()
	 * @generated
	 */
	int PROGRAM = 38;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROGRAM__NAME = 0;

	/**
	 * The feature id for the '<em><b>Tasks</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROGRAM__TASKS = 1;

	/**
	 * The feature id for the '<em><b>Functions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROGRAM__FUNCTIONS = 2;

	/**
	 * The feature id for the '<em><b>Subroutines</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROGRAM__SUBROUTINES = 3;

	/**
	 * The feature id for the '<em><b>Global Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROGRAM__GLOBAL_VARIABLES = 4;

	/**
	 * The number of structural features of the '<em>Program</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROGRAM_FEATURE_COUNT = 5;

	/**
	 * The meta object id for the '{@link nqc.impl.ProgramsImpl <em>Programs</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ProgramsImpl
	 * @see nqc.impl.NqcPackageImpl#getPrograms()
	 * @generated
	 */
	int PROGRAMS = 39;

	/**
	 * The feature id for the '<em><b>Programs</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROGRAMS__PROGRAMS = 0;

	/**
	 * The number of structural features of the '<em>Programs</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROGRAMS_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link nqc.impl.QuaternaryBuiltInFunctionCallImpl <em>Quaternary Built In Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.QuaternaryBuiltInFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getQuaternaryBuiltInFunctionCall()
	 * @generated
	 */
	int QUATERNARY_BUILT_IN_FUNCTION_CALL = 40;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int QUATERNARY_BUILT_IN_FUNCTION_CALL__LABEL = BUILT_IN_FUNCTION_CALL__LABEL;

	/**
	 * The feature id for the '<em><b>Quaternary Built In Function</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int QUATERNARY_BUILT_IN_FUNCTION_CALL__QUATERNARY_BUILT_IN_FUNCTION = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Parameter1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Parameter2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Parameter3</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Parameter4</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER4 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 4;

	/**
	 * The number of structural features of the '<em>Quaternary Built In Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int QUATERNARY_BUILT_IN_FUNCTION_CALL_FEATURE_COUNT = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 5;

	/**
	 * The meta object id for the '{@link nqc.impl.RepeatStatementImpl <em>Repeat Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.RepeatStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getRepeatStatement()
	 * @generated
	 */
	int REPEAT_STATEMENT = 41;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPEAT_STATEMENT__LABEL = CONTROL_STRUCTURE__LABEL;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPEAT_STATEMENT__CONDITION = CONTROL_STRUCTURE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Body</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPEAT_STATEMENT__BODY = CONTROL_STRUCTURE_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Repeat Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPEAT_STATEMENT_FEATURE_COUNT = CONTROL_STRUCTURE_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link nqc.impl.ReturnStatementImpl <em>Return Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.ReturnStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getReturnStatement()
	 * @generated
	 */
	int RETURN_STATEMENT = 42;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RETURN_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The number of structural features of the '<em>Return Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RETURN_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link nqc.impl.SenaryBuiltInFunctionCallImpl <em>Senary Built In Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SenaryBuiltInFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getSenaryBuiltInFunctionCall()
	 * @generated
	 */
	int SENARY_BUILT_IN_FUNCTION_CALL = 43;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENARY_BUILT_IN_FUNCTION_CALL__LABEL = BUILT_IN_FUNCTION_CALL__LABEL;

	/**
	 * The feature id for the '<em><b>Senary Built In Function</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENARY_BUILT_IN_FUNCTION_CALL__SENARY_BUILT_IN_FUNCTION = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Parameter1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENARY_BUILT_IN_FUNCTION_CALL__PARAMETER1 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Parameter2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENARY_BUILT_IN_FUNCTION_CALL__PARAMETER2 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Parameter3</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENARY_BUILT_IN_FUNCTION_CALL__PARAMETER3 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Parameter4</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENARY_BUILT_IN_FUNCTION_CALL__PARAMETER4 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Parameter5</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENARY_BUILT_IN_FUNCTION_CALL__PARAMETER5 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 5;

	/**
	 * The feature id for the '<em><b>Parameter6</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENARY_BUILT_IN_FUNCTION_CALL__PARAMETER6 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 6;

	/**
	 * The number of structural features of the '<em>Senary Built In Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENARY_BUILT_IN_FUNCTION_CALL_FEATURE_COUNT = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 7;

	/**
	 * The meta object id for the '{@link nqc.impl.SensorConfigConstantImpl <em>Sensor Config Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SensorConfigConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSensorConfigConstant()
	 * @generated
	 */
	int SENSOR_CONFIG_CONSTANT = 44;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_CONFIG_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Sensor Config</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_CONFIG_CONSTANT__SENSOR_CONFIG = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Sensor Config Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_CONFIG_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SensorModeConstantImpl <em>Sensor Mode Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SensorModeConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSensorModeConstant()
	 * @generated
	 */
	int SENSOR_MODE_CONSTANT = 45;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_MODE_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Sensor Mode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_MODE_CONSTANT__SENSOR_MODE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Sensor Mode Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_MODE_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SensorNameConstantImpl <em>Sensor Name Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SensorNameConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSensorNameConstant()
	 * @generated
	 */
	int SENSOR_NAME_CONSTANT = 46;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_NAME_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Sensor Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_NAME_CONSTANT__SENSOR_NAME = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Sensor Name Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_NAME_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SensorTypeConstantImpl <em>Sensor Type Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SensorTypeConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSensorTypeConstant()
	 * @generated
	 */
	int SENSOR_TYPE_CONSTANT = 47;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_TYPE_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Sensor Type</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_TYPE_CONSTANT__SENSOR_TYPE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Sensor Type Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_TYPE_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SerialBaudConstantImpl <em>Serial Baud Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SerialBaudConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSerialBaudConstant()
	 * @generated
	 */
	int SERIAL_BAUD_CONSTANT = 48;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_BAUD_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Serial Baud</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_BAUD_CONSTANT__SERIAL_BAUD = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Serial Baud Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_BAUD_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SerialBiphaseConstantImpl <em>Serial Biphase Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SerialBiphaseConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSerialBiphaseConstant()
	 * @generated
	 */
	int SERIAL_BIPHASE_CONSTANT = 49;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_BIPHASE_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Serial Biphase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_BIPHASE_CONSTANT__SERIAL_BIPHASE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Serial Biphase Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_BIPHASE_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SerialChecksumConstantImpl <em>Serial Checksum Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SerialChecksumConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSerialChecksumConstant()
	 * @generated
	 */
	int SERIAL_CHECKSUM_CONSTANT = 50;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_CHECKSUM_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Serial Checksum</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_CHECKSUM_CONSTANT__SERIAL_CHECKSUM = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Serial Checksum Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_CHECKSUM_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SerialChannelConstantImpl <em>Serial Channel Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SerialChannelConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSerialChannelConstant()
	 * @generated
	 */
	int SERIAL_CHANNEL_CONSTANT = 51;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_CHANNEL_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Serial Channel</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_CHANNEL_CONSTANT__SERIAL_CHANNEL = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Serial Channel Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_CHANNEL_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SerialCommConstantImpl <em>Serial Comm Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SerialCommConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSerialCommConstant()
	 * @generated
	 */
	int SERIAL_COMM_CONSTANT = 52;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_COMM_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Serial Comm</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_COMM_CONSTANT__SERIAL_COMM = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Serial Comm Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_COMM_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SerialPacketConstantImpl <em>Serial Packet Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SerialPacketConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSerialPacketConstant()
	 * @generated
	 */
	int SERIAL_PACKET_CONSTANT = 53;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_PACKET_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Serial Packet</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_PACKET_CONSTANT__SERIAL_PACKET = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Serial Packet Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERIAL_PACKET_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SoundConstantImpl <em>Sound Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SoundConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getSoundConstant()
	 * @generated
	 */
	int SOUND_CONSTANT = 54;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SOUND_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Sound</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SOUND_CONSTANT__SOUND = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Sound Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SOUND_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.StartStatementImpl <em>Start Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.StartStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getStartStatement()
	 * @generated
	 */
	int START_STATEMENT = 56;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int START_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The feature id for the '<em><b>Task</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int START_STATEMENT__TASK = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Start Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int START_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.StopStatementImpl <em>Stop Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.StopStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getStopStatement()
	 * @generated
	 */
	int STOP_STATEMENT = 57;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STOP_STATEMENT__LABEL = STATEMENT__LABEL;

	/**
	 * The feature id for the '<em><b>Task</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STOP_STATEMENT__TASK = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Stop Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STOP_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SubroutineImpl <em>Subroutine</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SubroutineImpl
	 * @see nqc.impl.NqcPackageImpl#getSubroutine()
	 * @generated
	 */
	int SUBROUTINE = 58;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SUBROUTINE__NAME = 0;

	/**
	 * The feature id for the '<em><b>Statements</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SUBROUTINE__STATEMENTS = 1;

	/**
	 * The feature id for the '<em><b>Local Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SUBROUTINE__LOCAL_VARIABLES = 2;

	/**
	 * The number of structural features of the '<em>Subroutine</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SUBROUTINE_FEATURE_COUNT = 3;

	/**
	 * The meta object id for the '{@link nqc.impl.SubroutineCallImpl <em>Subroutine Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SubroutineCallImpl
	 * @see nqc.impl.NqcPackageImpl#getSubroutineCall()
	 * @generated
	 */
	int SUBROUTINE_CALL = 59;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SUBROUTINE_CALL__LABEL = CALL_STATEMENT__LABEL;

	/**
	 * The feature id for the '<em><b>Callee</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SUBROUTINE_CALL__CALLEE = CALL_STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Subroutine Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SUBROUTINE_CALL_FEATURE_COUNT = CALL_STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.SwitchStatementImpl <em>Switch Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.SwitchStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getSwitchStatement()
	 * @generated
	 */
	int SWITCH_STATEMENT = 60;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SWITCH_STATEMENT__LABEL = CONTROL_STRUCTURE__LABEL;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SWITCH_STATEMENT__CONDITION = CONTROL_STRUCTURE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Cases</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SWITCH_STATEMENT__CASES = CONTROL_STRUCTURE_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Switch Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SWITCH_STATEMENT_FEATURE_COUNT = CONTROL_STRUCTURE_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link nqc.impl.TaskImpl <em>Task</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.TaskImpl
	 * @see nqc.impl.NqcPackageImpl#getTask()
	 * @generated
	 */
	int TASK = 61;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TASK__NAME = 0;

	/**
	 * The feature id for the '<em><b>Statements</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TASK__STATEMENTS = 1;

	/**
	 * The feature id for the '<em><b>Local Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TASK__LOCAL_VARIABLES = 2;

	/**
	 * The number of structural features of the '<em>Task</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TASK_FEATURE_COUNT = 3;

	/**
	 * The meta object id for the '{@link nqc.impl.TernaryBuiltInFunctionCallImpl <em>Ternary Built In Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.TernaryBuiltInFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getTernaryBuiltInFunctionCall()
	 * @generated
	 */
	int TERNARY_BUILT_IN_FUNCTION_CALL = 62;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_BUILT_IN_FUNCTION_CALL__LABEL = BUILT_IN_FUNCTION_CALL__LABEL;

	/**
	 * The feature id for the '<em><b>Ternary Built In Function</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_BUILT_IN_FUNCTION_CALL__TERNARY_BUILT_IN_FUNCTION = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Parameter1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Parameter2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Parameter3</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3 = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Ternary Built In Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_BUILT_IN_FUNCTION_CALL_FEATURE_COUNT = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 4;

	/**
	 * The meta object id for the '{@link nqc.impl.TernaryExpressionImpl <em>Ternary Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.TernaryExpressionImpl
	 * @see nqc.impl.NqcPackageImpl#getTernaryExpression()
	 * @generated
	 */
	int TERNARY_EXPRESSION = 63;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_EXPRESSION__LABEL = COMPOUND_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Operator</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_EXPRESSION__OPERATOR = COMPOUND_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Operand1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_EXPRESSION__OPERAND1 = COMPOUND_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Operand2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_EXPRESSION__OPERAND2 = COMPOUND_EXPRESSION_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Operand3</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_EXPRESSION__OPERAND3 = COMPOUND_EXPRESSION_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Ternary Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TERNARY_EXPRESSION_FEATURE_COUNT = COMPOUND_EXPRESSION_FEATURE_COUNT + 4;

	/**
	 * The meta object id for the '{@link nqc.impl.TxPowerConstantImpl <em>Tx Power Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.TxPowerConstantImpl
	 * @see nqc.impl.NqcPackageImpl#getTxPowerConstant()
	 * @generated
	 */
	int TX_POWER_CONSTANT = 64;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TX_POWER_CONSTANT__LABEL = CONSTANT_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Tx Power</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TX_POWER_CONSTANT__TX_POWER = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Tx Power Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TX_POWER_CONSTANT_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link nqc.impl.UnaryBuiltInFunctionCallImpl <em>Unary Built In Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.UnaryBuiltInFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getUnaryBuiltInFunctionCall()
	 * @generated
	 */
	int UNARY_BUILT_IN_FUNCTION_CALL = 65;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_BUILT_IN_FUNCTION_CALL__LABEL = BUILT_IN_FUNCTION_CALL__LABEL;

	/**
	 * The feature id for the '<em><b>Unary Built In Function</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_BUILT_IN_FUNCTION_CALL__UNARY_BUILT_IN_FUNCTION = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Parameter</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_BUILT_IN_FUNCTION_CALL__PARAMETER = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Unary Built In Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_BUILT_IN_FUNCTION_CALL_FEATURE_COUNT = BUILT_IN_FUNCTION_CALL_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link nqc.impl.UnaryBuiltInValueFunctionCallImpl <em>Unary Built In Value Function Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.UnaryBuiltInValueFunctionCallImpl
	 * @see nqc.impl.NqcPackageImpl#getUnaryBuiltInValueFunctionCall()
	 * @generated
	 */
	int UNARY_BUILT_IN_VALUE_FUNCTION_CALL = 66;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_BUILT_IN_VALUE_FUNCTION_CALL__LABEL = BUILT_IN_VALUE_FUNCTION_CALL__LABEL;

	/**
	 * The feature id for the '<em><b>Unary Built In Value Function</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_BUILT_IN_VALUE_FUNCTION_CALL__UNARY_BUILT_IN_VALUE_FUNCTION = BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Parameter</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER = BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Unary Built In Value Function Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT = BUILT_IN_VALUE_FUNCTION_CALL_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link nqc.impl.UnaryExpressionImpl <em>Unary Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.UnaryExpressionImpl
	 * @see nqc.impl.NqcPackageImpl#getUnaryExpression()
	 * @generated
	 */
	int UNARY_EXPRESSION = 67;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_EXPRESSION__LABEL = COMPOUND_EXPRESSION__LABEL;

	/**
	 * The feature id for the '<em><b>Operand</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_EXPRESSION__OPERAND = COMPOUND_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Operator</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_EXPRESSION__OPERATOR = COMPOUND_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Unary Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNARY_EXPRESSION_FEATURE_COUNT = COMPOUND_EXPRESSION_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link nqc.impl.UntilStatementImpl <em>Until Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.UntilStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getUntilStatement()
	 * @generated
	 */
	int UNTIL_STATEMENT = 68;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNTIL_STATEMENT__LABEL = CONTROL_STRUCTURE__LABEL;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNTIL_STATEMENT__CONDITION = CONTROL_STRUCTURE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Body</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNTIL_STATEMENT__BODY = CONTROL_STRUCTURE_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Until Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNTIL_STATEMENT_FEATURE_COUNT = CONTROL_STRUCTURE_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link nqc.impl.VariableImpl <em>Variable</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.VariableImpl
	 * @see nqc.impl.NqcPackageImpl#getVariable()
	 * @generated
	 */
	int VARIABLE = 70;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE__NAME = 0;

	/**
	 * The feature id for the '<em><b>Type</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE__TYPE = 1;

	/**
	 * The feature id for the '<em><b>Initial Value</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE__INITIAL_VALUE = 2;

	/**
	 * The feature id for the '<em><b>Array Max Size</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE__ARRAY_MAX_SIZE = 3;

	/**
	 * The number of structural features of the '<em>Variable</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE_FEATURE_COUNT = 4;

	/**
	 * The meta object id for the '{@link nqc.impl.WhileStatementImpl <em>While Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.impl.WhileStatementImpl
	 * @see nqc.impl.NqcPackageImpl#getWhileStatement()
	 * @generated
	 */
	int WHILE_STATEMENT = 72;

	/**
	 * The feature id for the '<em><b>Label</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int WHILE_STATEMENT__LABEL = CONTROL_STRUCTURE__LABEL;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int WHILE_STATEMENT__CONDITION = CONTROL_STRUCTURE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Body</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int WHILE_STATEMENT__BODY = CONTROL_STRUCTURE_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>While Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int WHILE_STATEMENT_FEATURE_COUNT = CONTROL_STRUCTURE_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link nqc.AcquireEnum <em>Acquire Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.AcquireEnum
	 * @see nqc.impl.NqcPackageImpl#getAcquireEnum()
	 * @generated
	 */
	int ACQUIRE_ENUM = 73;

	/**
	 * The meta object id for the '{@link nqc.AssignmentStatementEnum <em>Assignment Statement Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.AssignmentStatementEnum
	 * @see nqc.impl.NqcPackageImpl#getAssignmentStatementEnum()
	 * @generated
	 */
	int ASSIGNMENT_STATEMENT_ENUM = 74;

	/**
	 * The meta object id for the '{@link nqc.BinaryOperatorEnum <em>Binary Operator Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.BinaryOperatorEnum
	 * @see nqc.impl.NqcPackageImpl#getBinaryOperatorEnum()
	 * @generated
	 */
	int BINARY_OPERATOR_ENUM = 75;

	/**
	 * The meta object id for the '{@link nqc.BuiltInBinaryFunctionEnum <em>Built In Binary Function Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.BuiltInBinaryFunctionEnum
	 * @see nqc.impl.NqcPackageImpl#getBuiltInBinaryFunctionEnum()
	 * @generated
	 */
	int BUILT_IN_BINARY_FUNCTION_ENUM = 76;

	/**
	 * The meta object id for the '{@link nqc.BuiltInBinaryValueFunctionEnum <em>Built In Binary Value Function Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.BuiltInBinaryValueFunctionEnum
	 * @see nqc.impl.NqcPackageImpl#getBuiltInBinaryValueFunctionEnum()
	 * @generated
	 */
	int BUILT_IN_BINARY_VALUE_FUNCTION_ENUM = 77;

	/**
	 * The meta object id for the '{@link nqc.BuiltInNullaryFunctionEnum <em>Built In Nullary Function Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.BuiltInNullaryFunctionEnum
	 * @see nqc.impl.NqcPackageImpl#getBuiltInNullaryFunctionEnum()
	 * @generated
	 */
	int BUILT_IN_NULLARY_FUNCTION_ENUM = 78;

	/**
	 * The meta object id for the '{@link nqc.BuiltInNullaryValueFunctionEnum <em>Built In Nullary Value Function Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.BuiltInNullaryValueFunctionEnum
	 * @see nqc.impl.NqcPackageImpl#getBuiltInNullaryValueFunctionEnum()
	 * @generated
	 */
	int BUILT_IN_NULLARY_VALUE_FUNCTION_ENUM = 79;

	/**
	 * The meta object id for the '{@link nqc.BuiltInUnaryValueFunctionEnum <em>Built In Unary Value Function Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.BuiltInUnaryValueFunctionEnum
	 * @see nqc.impl.NqcPackageImpl#getBuiltInUnaryValueFunctionEnum()
	 * @generated
	 */
	int BUILT_IN_UNARY_VALUE_FUNCTION_ENUM = 80;

	/**
	 * The meta object id for the '{@link nqc.BuiltInUnaryFunctionEnum <em>Built In Unary Function Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.BuiltInUnaryFunctionEnum
	 * @see nqc.impl.NqcPackageImpl#getBuiltInUnaryFunctionEnum()
	 * @generated
	 */
	int BUILT_IN_UNARY_FUNCTION_ENUM = 81;

	/**
	 * The meta object id for the '{@link nqc.BuiltInQuaternaryFunctionEnum <em>Built In Quaternary Function Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.BuiltInQuaternaryFunctionEnum
	 * @see nqc.impl.NqcPackageImpl#getBuiltInQuaternaryFunctionEnum()
	 * @generated
	 */
	int BUILT_IN_QUATERNARY_FUNCTION_ENUM = 82;

	/**
	 * The meta object id for the '{@link nqc.BuiltInSenaryFunctionEnum <em>Built In Senary Function Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.BuiltInSenaryFunctionEnum
	 * @see nqc.impl.NqcPackageImpl#getBuiltInSenaryFunctionEnum()
	 * @generated
	 */
	int BUILT_IN_SENARY_FUNCTION_ENUM = 83;

	/**
	 * The meta object id for the '{@link nqc.BuiltInTernaryFunctionEnum <em>Built In Ternary Function Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.BuiltInTernaryFunctionEnum
	 * @see nqc.impl.NqcPackageImpl#getBuiltInTernaryFunctionEnum()
	 * @generated
	 */
	int BUILT_IN_TERNARY_FUNCTION_ENUM = 84;

	/**
	 * The meta object id for the '{@link nqc.DirectionEnum <em>Direction Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.DirectionEnum
	 * @see nqc.impl.NqcPackageImpl#getDirectionEnum()
	 * @generated
	 */
	int DIRECTION_ENUM = 85;

	/**
	 * The meta object id for the '{@link nqc.DisplayModeEnum <em>Display Mode Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.DisplayModeEnum
	 * @see nqc.impl.NqcPackageImpl#getDisplayModeEnum()
	 * @generated
	 */
	int DISPLAY_MODE_ENUM = 86;

	/**
	 * The meta object id for the '{@link nqc.EventTypeEnum <em>Event Type Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.EventTypeEnum
	 * @see nqc.impl.NqcPackageImpl#getEventTypeEnum()
	 * @generated
	 */
	int EVENT_TYPE_ENUM = 87;

	/**
	 * The meta object id for the '{@link nqc.OutputModeEnum <em>Output Mode Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.OutputModeEnum
	 * @see nqc.impl.NqcPackageImpl#getOutputModeEnum()
	 * @generated
	 */
	int OUTPUT_MODE_ENUM = 88;

	/**
	 * The meta object id for the '{@link nqc.OutputPortNameEnum <em>Output Port Name Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.OutputPortNameEnum
	 * @see nqc.impl.NqcPackageImpl#getOutputPortNameEnum()
	 * @generated
	 */
	int OUTPUT_PORT_NAME_ENUM = 89;

	/**
	 * The meta object id for the '{@link nqc.SerialBaudEnum <em>Serial Baud Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SerialBaudEnum
	 * @see nqc.impl.NqcPackageImpl#getSerialBaudEnum()
	 * @generated
	 */
	int SERIAL_BAUD_ENUM = 90;

	/**
	 * The meta object id for the '{@link nqc.SerialBiphaseEnum <em>Serial Biphase Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SerialBiphaseEnum
	 * @see nqc.impl.NqcPackageImpl#getSerialBiphaseEnum()
	 * @generated
	 */
	int SERIAL_BIPHASE_ENUM = 91;

	/**
	 * The meta object id for the '{@link nqc.SerialChannelEnum <em>Serial Channel Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SerialChannelEnum
	 * @see nqc.impl.NqcPackageImpl#getSerialChannelEnum()
	 * @generated
	 */
	int SERIAL_CHANNEL_ENUM = 92;

	/**
	 * The meta object id for the '{@link nqc.SerialChecksumEnum <em>Serial Checksum Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SerialChecksumEnum
	 * @see nqc.impl.NqcPackageImpl#getSerialChecksumEnum()
	 * @generated
	 */
	int SERIAL_CHECKSUM_ENUM = 93;

	/**
	 * The meta object id for the '{@link nqc.SerialCommEnum <em>Serial Comm Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SerialCommEnum
	 * @see nqc.impl.NqcPackageImpl#getSerialCommEnum()
	 * @generated
	 */
	int SERIAL_COMM_ENUM = 94;

	/**
	 * The meta object id for the '{@link nqc.SerialPacketEnum <em>Serial Packet Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SerialPacketEnum
	 * @see nqc.impl.NqcPackageImpl#getSerialPacketEnum()
	 * @generated
	 */
	int SERIAL_PACKET_ENUM = 95;

	/**
	 * The meta object id for the '{@link nqc.SensorConfigEnum <em>Sensor Config Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SensorConfigEnum
	 * @see nqc.impl.NqcPackageImpl#getSensorConfigEnum()
	 * @generated
	 */
	int SENSOR_CONFIG_ENUM = 96;

	/**
	 * The meta object id for the '{@link nqc.SensorModeEnum <em>Sensor Mode Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SensorModeEnum
	 * @see nqc.impl.NqcPackageImpl#getSensorModeEnum()
	 * @generated
	 */
	int SENSOR_MODE_ENUM = 97;

	/**
	 * The meta object id for the '{@link nqc.SensorNameEnum <em>Sensor Name Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SensorNameEnum
	 * @see nqc.impl.NqcPackageImpl#getSensorNameEnum()
	 * @generated
	 */
	int SENSOR_NAME_ENUM = 98;

	/**
	 * The meta object id for the '{@link nqc.SensorTypeEnum <em>Sensor Type Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SensorTypeEnum
	 * @see nqc.impl.NqcPackageImpl#getSensorTypeEnum()
	 * @generated
	 */
	int SENSOR_TYPE_ENUM = 99;

	/**
	 * The meta object id for the '{@link nqc.SoundEnum <em>Sound Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.SoundEnum
	 * @see nqc.impl.NqcPackageImpl#getSoundEnum()
	 * @generated
	 */
	int SOUND_ENUM = 100;

	/**
	 * The meta object id for the '{@link nqc.TernaryOperatorEnum <em>Ternary Operator Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.TernaryOperatorEnum
	 * @see nqc.impl.NqcPackageImpl#getTernaryOperatorEnum()
	 * @generated
	 */
	int TERNARY_OPERATOR_ENUM = 101;

	/**
	 * The meta object id for the '{@link nqc.TxPowerEnum <em>Tx Power Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.TxPowerEnum
	 * @see nqc.impl.NqcPackageImpl#getTxPowerEnum()
	 * @generated
	 */
	int TX_POWER_ENUM = 102;

	/**
	 * The meta object id for the '{@link nqc.TypeEnum <em>Type Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.TypeEnum
	 * @see nqc.impl.NqcPackageImpl#getTypeEnum()
	 * @generated
	 */
	int TYPE_ENUM = 103;

	/**
	 * The meta object id for the '{@link nqc.UnaryOperatorEnum <em>Unary Operator Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see nqc.UnaryOperatorEnum
	 * @see nqc.impl.NqcPackageImpl#getUnaryOperatorEnum()
	 * @generated
	 */
	int UNARY_OPERATOR_ENUM = 104;


	/**
	 * Returns the meta object for class '{@link nqc.AcquireConstant <em>Acquire Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Acquire Constant</em>'.
	 * @see nqc.AcquireConstant
	 * @generated
	 */
	EClass getAcquireConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.AcquireConstant#getAcquire <em>Acquire</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Acquire</em>'.
	 * @see nqc.AcquireConstant#getAcquire()
	 * @see #getAcquireConstant()
	 * @generated
	 */
	EAttribute getAcquireConstant_Acquire();

	/**
	 * Returns the meta object for class '{@link nqc.AcquireStatement <em>Acquire Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Acquire Statement</em>'.
	 * @see nqc.AcquireStatement
	 * @generated
	 */
	EClass getAcquireStatement();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.AcquireStatement#getResource <em>Resource</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Resource</em>'.
	 * @see nqc.AcquireStatement#getResource()
	 * @see #getAcquireStatement()
	 * @generated
	 */
	EReference getAcquireStatement_Resource();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.AcquireStatement#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Body</em>'.
	 * @see nqc.AcquireStatement#getBody()
	 * @see #getAcquireStatement()
	 * @generated
	 */
	EReference getAcquireStatement_Body();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.AcquireStatement#getHandler <em>Handler</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Handler</em>'.
	 * @see nqc.AcquireStatement#getHandler()
	 * @see #getAcquireStatement()
	 * @generated
	 */
	EReference getAcquireStatement_Handler();

	/**
	 * Returns the meta object for class '{@link nqc.ArrayExpression <em>Array Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Array Expression</em>'.
	 * @see nqc.ArrayExpression
	 * @generated
	 */
	EClass getArrayExpression();

	/**
	 * Returns the meta object for the reference '{@link nqc.ArrayExpression#getIndex <em>Index</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Index</em>'.
	 * @see nqc.ArrayExpression#getIndex()
	 * @see #getArrayExpression()
	 * @generated
	 */
	EReference getArrayExpression_Index();

	/**
	 * Returns the meta object for class '{@link nqc.AssignmentStatement <em>Assignment Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Assignment Statement</em>'.
	 * @see nqc.AssignmentStatement
	 * @generated
	 */
	EClass getAssignmentStatement();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.AssignmentStatement#getVariable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Variable</em>'.
	 * @see nqc.AssignmentStatement#getVariable()
	 * @see #getAssignmentStatement()
	 * @generated
	 */
	EReference getAssignmentStatement_Variable();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.AssignmentStatement#getExpression <em>Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Expression</em>'.
	 * @see nqc.AssignmentStatement#getExpression()
	 * @see #getAssignmentStatement()
	 * @generated
	 */
	EReference getAssignmentStatement_Expression();

	/**
	 * Returns the meta object for the attribute '{@link nqc.AssignmentStatement#getOperator <em>Operator</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Operator</em>'.
	 * @see nqc.AssignmentStatement#getOperator()
	 * @see #getAssignmentStatement()
	 * @generated
	 */
	EAttribute getAssignmentStatement_Operator();

	/**
	 * Returns the meta object for class '{@link nqc.BinaryExpression <em>Binary Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Binary Expression</em>'.
	 * @see nqc.BinaryExpression
	 * @generated
	 */
	EClass getBinaryExpression();

	/**
	 * Returns the meta object for the attribute '{@link nqc.BinaryExpression#getOperator <em>Operator</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Operator</em>'.
	 * @see nqc.BinaryExpression#getOperator()
	 * @see #getBinaryExpression()
	 * @generated
	 */
	EAttribute getBinaryExpression_Operator();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.BinaryExpression#getOperand1 <em>Operand1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand1</em>'.
	 * @see nqc.BinaryExpression#getOperand1()
	 * @see #getBinaryExpression()
	 * @generated
	 */
	EReference getBinaryExpression_Operand1();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.BinaryExpression#getOperand2 <em>Operand2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand2</em>'.
	 * @see nqc.BinaryExpression#getOperand2()
	 * @see #getBinaryExpression()
	 * @generated
	 */
	EReference getBinaryExpression_Operand2();

	/**
	 * Returns the meta object for class '{@link nqc.BinaryBuiltInFunctionCall <em>Binary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Binary Built In Function Call</em>'.
	 * @see nqc.BinaryBuiltInFunctionCall
	 * @generated
	 */
	EClass getBinaryBuiltInFunctionCall();

	/**
	 * Returns the meta object for the attribute '{@link nqc.BinaryBuiltInFunctionCall#getBinaryBuiltInFunction <em>Binary Built In Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Binary Built In Function</em>'.
	 * @see nqc.BinaryBuiltInFunctionCall#getBinaryBuiltInFunction()
	 * @see #getBinaryBuiltInFunctionCall()
	 * @generated
	 */
	EAttribute getBinaryBuiltInFunctionCall_BinaryBuiltInFunction();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.BinaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter1</em>'.
	 * @see nqc.BinaryBuiltInFunctionCall#getParameter1()
	 * @see #getBinaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getBinaryBuiltInFunctionCall_Parameter1();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.BinaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter2</em>'.
	 * @see nqc.BinaryBuiltInFunctionCall#getParameter2()
	 * @see #getBinaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getBinaryBuiltInFunctionCall_Parameter2();

	/**
	 * Returns the meta object for class '{@link nqc.BinaryBuiltInValueFunctionCall <em>Binary Built In Value Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Binary Built In Value Function Call</em>'.
	 * @see nqc.BinaryBuiltInValueFunctionCall
	 * @generated
	 */
	EClass getBinaryBuiltInValueFunctionCall();

	/**
	 * Returns the meta object for the attribute '{@link nqc.BinaryBuiltInValueFunctionCall#getBinaryBuiltInValueFunction <em>Binary Built In Value Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Binary Built In Value Function</em>'.
	 * @see nqc.BinaryBuiltInValueFunctionCall#getBinaryBuiltInValueFunction()
	 * @see #getBinaryBuiltInValueFunctionCall()
	 * @generated
	 */
	EAttribute getBinaryBuiltInValueFunctionCall_BinaryBuiltInValueFunction();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.BinaryBuiltInValueFunctionCall#getParameter1 <em>Parameter1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter1</em>'.
	 * @see nqc.BinaryBuiltInValueFunctionCall#getParameter1()
	 * @see #getBinaryBuiltInValueFunctionCall()
	 * @generated
	 */
	EReference getBinaryBuiltInValueFunctionCall_Parameter1();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.BinaryBuiltInValueFunctionCall#getParameter2 <em>Parameter2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter2</em>'.
	 * @see nqc.BinaryBuiltInValueFunctionCall#getParameter2()
	 * @see #getBinaryBuiltInValueFunctionCall()
	 * @generated
	 */
	EReference getBinaryBuiltInValueFunctionCall_Parameter2();

	/**
	 * Returns the meta object for class '{@link nqc.BlockStatement <em>Block Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Block Statement</em>'.
	 * @see nqc.BlockStatement
	 * @generated
	 */
	EClass getBlockStatement();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.BlockStatement#getLocalVariables <em>Local Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Local Variables</em>'.
	 * @see nqc.BlockStatement#getLocalVariables()
	 * @see #getBlockStatement()
	 * @generated
	 */
	EReference getBlockStatement_LocalVariables();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.BlockStatement#getStatements <em>Statements</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Statements</em>'.
	 * @see nqc.BlockStatement#getStatements()
	 * @see #getBlockStatement()
	 * @generated
	 */
	EReference getBlockStatement_Statements();

	/**
	 * Returns the meta object for class '{@link nqc.BooleanConstant <em>Boolean Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Boolean Constant</em>'.
	 * @see nqc.BooleanConstant
	 * @generated
	 */
	EClass getBooleanConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.BooleanConstant#isValue <em>Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Value</em>'.
	 * @see nqc.BooleanConstant#isValue()
	 * @see #getBooleanConstant()
	 * @generated
	 */
	EAttribute getBooleanConstant_Value();

	/**
	 * Returns the meta object for class '{@link nqc.BreakStatement <em>Break Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Break Statement</em>'.
	 * @see nqc.BreakStatement
	 * @generated
	 */
	EClass getBreakStatement();

	/**
	 * Returns the meta object for class '{@link nqc.BuiltInFunctionCall <em>Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Built In Function Call</em>'.
	 * @see nqc.BuiltInFunctionCall
	 * @generated
	 */
	EClass getBuiltInFunctionCall();

	/**
	 * Returns the meta object for class '{@link nqc.BuiltInValueFunctionCall <em>Built In Value Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Built In Value Function Call</em>'.
	 * @see nqc.BuiltInValueFunctionCall
	 * @generated
	 */
	EClass getBuiltInValueFunctionCall();

	/**
	 * Returns the meta object for class '{@link nqc.CallStatement <em>Call Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Call Statement</em>'.
	 * @see nqc.CallStatement
	 * @generated
	 */
	EClass getCallStatement();

	/**
	 * Returns the meta object for class '{@link nqc.Case <em>Case</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Case</em>'.
	 * @see nqc.Case
	 * @generated
	 */
	EClass getCase();

	/**
	 * Returns the meta object for the attribute '{@link nqc.Case#isIsDefault <em>Is Default</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Is Default</em>'.
	 * @see nqc.Case#isIsDefault()
	 * @see #getCase()
	 * @generated
	 */
	EAttribute getCase_IsDefault();

	/**
	 * Returns the meta object for the reference list '{@link nqc.Case#getLabel <em>Label</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Label</em>'.
	 * @see nqc.Case#getLabel()
	 * @see #getCase()
	 * @generated
	 */
	EReference getCase_Label();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Case#getStatements <em>Statements</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Statements</em>'.
	 * @see nqc.Case#getStatements()
	 * @see #getCase()
	 * @generated
	 */
	EReference getCase_Statements();

	/**
	 * Returns the meta object for class '{@link nqc.CompoundExpression <em>Compound Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Compound Expression</em>'.
	 * @see nqc.CompoundExpression
	 * @generated
	 */
	EClass getCompoundExpression();

	/**
	 * Returns the meta object for class '{@link nqc.ConstantExpression <em>Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Constant Expression</em>'.
	 * @see nqc.ConstantExpression
	 * @generated
	 */
	EClass getConstantExpression();

	/**
	 * Returns the meta object for class '{@link nqc.ContinueStatement <em>Continue Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Continue Statement</em>'.
	 * @see nqc.ContinueStatement
	 * @generated
	 */
	EClass getContinueStatement();

	/**
	 * Returns the meta object for class '{@link nqc.ControlStructure <em>Control Structure</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Control Structure</em>'.
	 * @see nqc.ControlStructure
	 * @generated
	 */
	EClass getControlStructure();

	/**
	 * Returns the meta object for class '{@link nqc.DirectionConstant <em>Direction Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Direction Constant</em>'.
	 * @see nqc.DirectionConstant
	 * @generated
	 */
	EClass getDirectionConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.DirectionConstant#getDirection <em>Direction</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Direction</em>'.
	 * @see nqc.DirectionConstant#getDirection()
	 * @see #getDirectionConstant()
	 * @generated
	 */
	EAttribute getDirectionConstant_Direction();

	/**
	 * Returns the meta object for class '{@link nqc.DisplayModeConstant <em>Display Mode Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Display Mode Constant</em>'.
	 * @see nqc.DisplayModeConstant
	 * @generated
	 */
	EClass getDisplayModeConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.DisplayModeConstant#getDisplayMode <em>Display Mode</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Display Mode</em>'.
	 * @see nqc.DisplayModeConstant#getDisplayMode()
	 * @see #getDisplayModeConstant()
	 * @generated
	 */
	EAttribute getDisplayModeConstant_DisplayMode();

	/**
	 * Returns the meta object for class '{@link nqc.DoWhileStatement <em>Do While Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Do While Statement</em>'.
	 * @see nqc.DoWhileStatement
	 * @generated
	 */
	EClass getDoWhileStatement();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.DoWhileStatement#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Body</em>'.
	 * @see nqc.DoWhileStatement#getBody()
	 * @see #getDoWhileStatement()
	 * @generated
	 */
	EReference getDoWhileStatement_Body();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.DoWhileStatement#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Condition</em>'.
	 * @see nqc.DoWhileStatement#getCondition()
	 * @see #getDoWhileStatement()
	 * @generated
	 */
	EReference getDoWhileStatement_Condition();

	/**
	 * Returns the meta object for class '{@link nqc.EmptyStatement <em>Empty Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Empty Statement</em>'.
	 * @see nqc.EmptyStatement
	 * @generated
	 */
	EClass getEmptyStatement();

	/**
	 * Returns the meta object for class '{@link nqc.EventTypeConstant <em>Event Type Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Event Type Constant</em>'.
	 * @see nqc.EventTypeConstant
	 * @generated
	 */
	EClass getEventTypeConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.EventTypeConstant#getEventType <em>Event Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Event Type</em>'.
	 * @see nqc.EventTypeConstant#getEventType()
	 * @see #getEventTypeConstant()
	 * @generated
	 */
	EAttribute getEventTypeConstant_EventType();

	/**
	 * Returns the meta object for class '{@link nqc.Expression <em>Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Expression</em>'.
	 * @see nqc.Expression
	 * @generated
	 */
	EClass getExpression();

	/**
	 * Returns the meta object for class '{@link nqc.Function <em>Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Function</em>'.
	 * @see nqc.Function
	 * @generated
	 */
	EClass getFunction();

	/**
	 * Returns the meta object for the attribute '{@link nqc.Function#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see nqc.Function#getName()
	 * @see #getFunction()
	 * @generated
	 */
	EAttribute getFunction_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Function#getStatements <em>Statements</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Statements</em>'.
	 * @see nqc.Function#getStatements()
	 * @see #getFunction()
	 * @generated
	 */
	EReference getFunction_Statements();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Function#getLocalVariables <em>Local Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Local Variables</em>'.
	 * @see nqc.Function#getLocalVariables()
	 * @see #getFunction()
	 * @generated
	 */
	EReference getFunction_LocalVariables();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Function#getParameters <em>Parameters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Parameters</em>'.
	 * @see nqc.Function#getParameters()
	 * @see #getFunction()
	 * @generated
	 */
	EReference getFunction_Parameters();

	/**
	 * Returns the meta object for class '{@link nqc.FunctionCall <em>Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Function Call</em>'.
	 * @see nqc.FunctionCall
	 * @generated
	 */
	EClass getFunctionCall();

	/**
	 * Returns the meta object for the reference '{@link nqc.FunctionCall#getCallee <em>Callee</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Callee</em>'.
	 * @see nqc.FunctionCall#getCallee()
	 * @see #getFunctionCall()
	 * @generated
	 */
	EReference getFunctionCall_Callee();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.FunctionCall#getParameters <em>Parameters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Parameters</em>'.
	 * @see nqc.FunctionCall#getParameters()
	 * @see #getFunctionCall()
	 * @generated
	 */
	EReference getFunctionCall_Parameters();

	/**
	 * Returns the meta object for class '{@link nqc.ForStatement <em>For Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>For Statement</em>'.
	 * @see nqc.ForStatement
	 * @generated
	 */
	EClass getForStatement();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.ForStatement#getInitialization <em>Initialization</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Initialization</em>'.
	 * @see nqc.ForStatement#getInitialization()
	 * @see #getForStatement()
	 * @generated
	 */
	EReference getForStatement_Initialization();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.ForStatement#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Condition</em>'.
	 * @see nqc.ForStatement#getCondition()
	 * @see #getForStatement()
	 * @generated
	 */
	EReference getForStatement_Condition();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.ForStatement#getUpdate <em>Update</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Update</em>'.
	 * @see nqc.ForStatement#getUpdate()
	 * @see #getForStatement()
	 * @generated
	 */
	EReference getForStatement_Update();

	/**
	 * Returns the meta object for class '{@link nqc.GoToStatement <em>Go To Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Go To Statement</em>'.
	 * @see nqc.GoToStatement
	 * @generated
	 */
	EClass getGoToStatement();

	/**
	 * Returns the meta object for the reference '{@link nqc.GoToStatement#getJumpLabel <em>Jump Label</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Jump Label</em>'.
	 * @see nqc.GoToStatement#getJumpLabel()
	 * @see #getGoToStatement()
	 * @generated
	 */
	EReference getGoToStatement_JumpLabel();

	/**
	 * Returns the meta object for class '{@link nqc.IfStatement <em>If Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>If Statement</em>'.
	 * @see nqc.IfStatement
	 * @generated
	 */
	EClass getIfStatement();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.IfStatement#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Condition</em>'.
	 * @see nqc.IfStatement#getCondition()
	 * @see #getIfStatement()
	 * @generated
	 */
	EReference getIfStatement_Condition();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.IfStatement#getConsequence <em>Consequence</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Consequence</em>'.
	 * @see nqc.IfStatement#getConsequence()
	 * @see #getIfStatement()
	 * @generated
	 */
	EReference getIfStatement_Consequence();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.IfStatement#getAlternative <em>Alternative</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Alternative</em>'.
	 * @see nqc.IfStatement#getAlternative()
	 * @see #getIfStatement()
	 * @generated
	 */
	EReference getIfStatement_Alternative();

	/**
	 * Returns the meta object for class '{@link nqc.IntegerConstant <em>Integer Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Integer Constant</em>'.
	 * @see nqc.IntegerConstant
	 * @generated
	 */
	EClass getIntegerConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.IntegerConstant#getValue <em>Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Value</em>'.
	 * @see nqc.IntegerConstant#getValue()
	 * @see #getIntegerConstant()
	 * @generated
	 */
	EAttribute getIntegerConstant_Value();

	/**
	 * Returns the meta object for class '{@link nqc.Label <em>Label</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Label</em>'.
	 * @see nqc.Label
	 * @generated
	 */
	EClass getLabel();

	/**
	 * Returns the meta object for the attribute '{@link nqc.Label#getLabel <em>Label</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Label</em>'.
	 * @see nqc.Label#getLabel()
	 * @see #getLabel()
	 * @generated
	 */
	EAttribute getLabel_Label();

	/**
	 * Returns the meta object for class '{@link nqc.MonitorHandler <em>Monitor Handler</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Monitor Handler</em>'.
	 * @see nqc.MonitorHandler
	 * @generated
	 */
	EClass getMonitorHandler();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.MonitorHandler#getCatchEvent <em>Catch Event</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Catch Event</em>'.
	 * @see nqc.MonitorHandler#getCatchEvent()
	 * @see #getMonitorHandler()
	 * @generated
	 */
	EReference getMonitorHandler_CatchEvent();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.MonitorHandler#getHandler <em>Handler</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Handler</em>'.
	 * @see nqc.MonitorHandler#getHandler()
	 * @see #getMonitorHandler()
	 * @generated
	 */
	EReference getMonitorHandler_Handler();

	/**
	 * Returns the meta object for class '{@link nqc.MonitorStatement <em>Monitor Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Monitor Statement</em>'.
	 * @see nqc.MonitorStatement
	 * @generated
	 */
	EClass getMonitorStatement();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.MonitorStatement#getEvents <em>Events</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Events</em>'.
	 * @see nqc.MonitorStatement#getEvents()
	 * @see #getMonitorStatement()
	 * @generated
	 */
	EReference getMonitorStatement_Events();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.MonitorStatement#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Body</em>'.
	 * @see nqc.MonitorStatement#getBody()
	 * @see #getMonitorStatement()
	 * @generated
	 */
	EReference getMonitorStatement_Body();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.MonitorStatement#getHandlers <em>Handlers</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Handlers</em>'.
	 * @see nqc.MonitorStatement#getHandlers()
	 * @see #getMonitorStatement()
	 * @generated
	 */
	EReference getMonitorStatement_Handlers();

	/**
	 * Returns the meta object for class '{@link nqc.NullaryBuiltInFunctionCall <em>Nullary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Nullary Built In Function Call</em>'.
	 * @see nqc.NullaryBuiltInFunctionCall
	 * @generated
	 */
	EClass getNullaryBuiltInFunctionCall();

	/**
	 * Returns the meta object for the attribute '{@link nqc.NullaryBuiltInFunctionCall#getNullaryBuiltInFunction <em>Nullary Built In Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Nullary Built In Function</em>'.
	 * @see nqc.NullaryBuiltInFunctionCall#getNullaryBuiltInFunction()
	 * @see #getNullaryBuiltInFunctionCall()
	 * @generated
	 */
	EAttribute getNullaryBuiltInFunctionCall_NullaryBuiltInFunction();

	/**
	 * Returns the meta object for class '{@link nqc.NullaryBuiltInValueFunctionCall <em>Nullary Built In Value Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Nullary Built In Value Function Call</em>'.
	 * @see nqc.NullaryBuiltInValueFunctionCall
	 * @generated
	 */
	EClass getNullaryBuiltInValueFunctionCall();

	/**
	 * Returns the meta object for the attribute '{@link nqc.NullaryBuiltInValueFunctionCall#getNullaryBuiltInValueFunction <em>Nullary Built In Value Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Nullary Built In Value Function</em>'.
	 * @see nqc.NullaryBuiltInValueFunctionCall#getNullaryBuiltInValueFunction()
	 * @see #getNullaryBuiltInValueFunctionCall()
	 * @generated
	 */
	EAttribute getNullaryBuiltInValueFunctionCall_NullaryBuiltInValueFunction();

	/**
	 * Returns the meta object for class '{@link nqc.OutputModeConstant <em>Output Mode Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Output Mode Constant</em>'.
	 * @see nqc.OutputModeConstant
	 * @generated
	 */
	EClass getOutputModeConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.OutputModeConstant#getOutputMode <em>Output Mode</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Output Mode</em>'.
	 * @see nqc.OutputModeConstant#getOutputMode()
	 * @see #getOutputModeConstant()
	 * @generated
	 */
	EAttribute getOutputModeConstant_OutputMode();

	/**
	 * Returns the meta object for class '{@link nqc.OutputPortNameConstant <em>Output Port Name Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Output Port Name Constant</em>'.
	 * @see nqc.OutputPortNameConstant
	 * @generated
	 */
	EClass getOutputPortNameConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.OutputPortNameConstant#getOutputPortName <em>Output Port Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Output Port Name</em>'.
	 * @see nqc.OutputPortNameConstant#getOutputPortName()
	 * @see #getOutputPortNameConstant()
	 * @generated
	 */
	EAttribute getOutputPortNameConstant_OutputPortName();

	/**
	 * Returns the meta object for class '{@link nqc.Parameter <em>Parameter</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Parameter</em>'.
	 * @see nqc.Parameter
	 * @generated
	 */
	EClass getParameter();

	/**
	 * Returns the meta object for the attribute '{@link nqc.Parameter#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see nqc.Parameter#getName()
	 * @see #getParameter()
	 * @generated
	 */
	EAttribute getParameter_Name();

	/**
	 * Returns the meta object for the attribute '{@link nqc.Parameter#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Type</em>'.
	 * @see nqc.Parameter#getType()
	 * @see #getParameter()
	 * @generated
	 */
	EAttribute getParameter_Type();

	/**
	 * Returns the meta object for class '{@link nqc.Program <em>Program</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Program</em>'.
	 * @see nqc.Program
	 * @generated
	 */
	EClass getProgram();

	/**
	 * Returns the meta object for the attribute '{@link nqc.Program#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see nqc.Program#getName()
	 * @see #getProgram()
	 * @generated
	 */
	EAttribute getProgram_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Program#getTasks <em>Tasks</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Tasks</em>'.
	 * @see nqc.Program#getTasks()
	 * @see #getProgram()
	 * @generated
	 */
	EReference getProgram_Tasks();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Program#getFunctions <em>Functions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Functions</em>'.
	 * @see nqc.Program#getFunctions()
	 * @see #getProgram()
	 * @generated
	 */
	EReference getProgram_Functions();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Program#getSubroutines <em>Subroutines</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Subroutines</em>'.
	 * @see nqc.Program#getSubroutines()
	 * @see #getProgram()
	 * @generated
	 */
	EReference getProgram_Subroutines();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Program#getGlobalVariables <em>Global Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Global Variables</em>'.
	 * @see nqc.Program#getGlobalVariables()
	 * @see #getProgram()
	 * @generated
	 */
	EReference getProgram_GlobalVariables();

	/**
	 * Returns the meta object for class '{@link nqc.Programs <em>Programs</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Programs</em>'.
	 * @see nqc.Programs
	 * @generated
	 */
	EClass getPrograms();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Programs#getPrograms <em>Programs</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Programs</em>'.
	 * @see nqc.Programs#getPrograms()
	 * @see #getPrograms()
	 * @generated
	 */
	EReference getPrograms_Programs();

	/**
	 * Returns the meta object for class '{@link nqc.QuaternaryBuiltInFunctionCall <em>Quaternary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Quaternary Built In Function Call</em>'.
	 * @see nqc.QuaternaryBuiltInFunctionCall
	 * @generated
	 */
	EClass getQuaternaryBuiltInFunctionCall();

	/**
	 * Returns the meta object for the attribute '{@link nqc.QuaternaryBuiltInFunctionCall#getQuaternaryBuiltInFunction <em>Quaternary Built In Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Quaternary Built In Function</em>'.
	 * @see nqc.QuaternaryBuiltInFunctionCall#getQuaternaryBuiltInFunction()
	 * @see #getQuaternaryBuiltInFunctionCall()
	 * @generated
	 */
	EAttribute getQuaternaryBuiltInFunctionCall_QuaternaryBuiltInFunction();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.QuaternaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter1</em>'.
	 * @see nqc.QuaternaryBuiltInFunctionCall#getParameter1()
	 * @see #getQuaternaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getQuaternaryBuiltInFunctionCall_Parameter1();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.QuaternaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter2</em>'.
	 * @see nqc.QuaternaryBuiltInFunctionCall#getParameter2()
	 * @see #getQuaternaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getQuaternaryBuiltInFunctionCall_Parameter2();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.QuaternaryBuiltInFunctionCall#getParameter3 <em>Parameter3</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter3</em>'.
	 * @see nqc.QuaternaryBuiltInFunctionCall#getParameter3()
	 * @see #getQuaternaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getQuaternaryBuiltInFunctionCall_Parameter3();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.QuaternaryBuiltInFunctionCall#getParameter4 <em>Parameter4</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter4</em>'.
	 * @see nqc.QuaternaryBuiltInFunctionCall#getParameter4()
	 * @see #getQuaternaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getQuaternaryBuiltInFunctionCall_Parameter4();

	/**
	 * Returns the meta object for class '{@link nqc.RepeatStatement <em>Repeat Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Repeat Statement</em>'.
	 * @see nqc.RepeatStatement
	 * @generated
	 */
	EClass getRepeatStatement();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.RepeatStatement#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Condition</em>'.
	 * @see nqc.RepeatStatement#getCondition()
	 * @see #getRepeatStatement()
	 * @generated
	 */
	EReference getRepeatStatement_Condition();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.RepeatStatement#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Body</em>'.
	 * @see nqc.RepeatStatement#getBody()
	 * @see #getRepeatStatement()
	 * @generated
	 */
	EReference getRepeatStatement_Body();

	/**
	 * Returns the meta object for class '{@link nqc.ReturnStatement <em>Return Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Return Statement</em>'.
	 * @see nqc.ReturnStatement
	 * @generated
	 */
	EClass getReturnStatement();

	/**
	 * Returns the meta object for class '{@link nqc.SenaryBuiltInFunctionCall <em>Senary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Senary Built In Function Call</em>'.
	 * @see nqc.SenaryBuiltInFunctionCall
	 * @generated
	 */
	EClass getSenaryBuiltInFunctionCall();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SenaryBuiltInFunctionCall#getSenaryBuiltInFunction <em>Senary Built In Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Senary Built In Function</em>'.
	 * @see nqc.SenaryBuiltInFunctionCall#getSenaryBuiltInFunction()
	 * @see #getSenaryBuiltInFunctionCall()
	 * @generated
	 */
	EAttribute getSenaryBuiltInFunctionCall_SenaryBuiltInFunction();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.SenaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter1</em>'.
	 * @see nqc.SenaryBuiltInFunctionCall#getParameter1()
	 * @see #getSenaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getSenaryBuiltInFunctionCall_Parameter1();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.SenaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter2</em>'.
	 * @see nqc.SenaryBuiltInFunctionCall#getParameter2()
	 * @see #getSenaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getSenaryBuiltInFunctionCall_Parameter2();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.SenaryBuiltInFunctionCall#getParameter3 <em>Parameter3</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter3</em>'.
	 * @see nqc.SenaryBuiltInFunctionCall#getParameter3()
	 * @see #getSenaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getSenaryBuiltInFunctionCall_Parameter3();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.SenaryBuiltInFunctionCall#getParameter4 <em>Parameter4</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter4</em>'.
	 * @see nqc.SenaryBuiltInFunctionCall#getParameter4()
	 * @see #getSenaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getSenaryBuiltInFunctionCall_Parameter4();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.SenaryBuiltInFunctionCall#getParameter5 <em>Parameter5</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter5</em>'.
	 * @see nqc.SenaryBuiltInFunctionCall#getParameter5()
	 * @see #getSenaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getSenaryBuiltInFunctionCall_Parameter5();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.SenaryBuiltInFunctionCall#getParameter6 <em>Parameter6</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter6</em>'.
	 * @see nqc.SenaryBuiltInFunctionCall#getParameter6()
	 * @see #getSenaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getSenaryBuiltInFunctionCall_Parameter6();

	/**
	 * Returns the meta object for class '{@link nqc.SensorConfigConstant <em>Sensor Config Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Sensor Config Constant</em>'.
	 * @see nqc.SensorConfigConstant
	 * @generated
	 */
	EClass getSensorConfigConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SensorConfigConstant#getSensorConfig <em>Sensor Config</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Sensor Config</em>'.
	 * @see nqc.SensorConfigConstant#getSensorConfig()
	 * @see #getSensorConfigConstant()
	 * @generated
	 */
	EAttribute getSensorConfigConstant_SensorConfig();

	/**
	 * Returns the meta object for class '{@link nqc.SensorModeConstant <em>Sensor Mode Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Sensor Mode Constant</em>'.
	 * @see nqc.SensorModeConstant
	 * @generated
	 */
	EClass getSensorModeConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SensorModeConstant#getSensorMode <em>Sensor Mode</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Sensor Mode</em>'.
	 * @see nqc.SensorModeConstant#getSensorMode()
	 * @see #getSensorModeConstant()
	 * @generated
	 */
	EAttribute getSensorModeConstant_SensorMode();

	/**
	 * Returns the meta object for class '{@link nqc.SensorNameConstant <em>Sensor Name Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Sensor Name Constant</em>'.
	 * @see nqc.SensorNameConstant
	 * @generated
	 */
	EClass getSensorNameConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SensorNameConstant#getSensorName <em>Sensor Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Sensor Name</em>'.
	 * @see nqc.SensorNameConstant#getSensorName()
	 * @see #getSensorNameConstant()
	 * @generated
	 */
	EAttribute getSensorNameConstant_SensorName();

	/**
	 * Returns the meta object for class '{@link nqc.SensorTypeConstant <em>Sensor Type Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Sensor Type Constant</em>'.
	 * @see nqc.SensorTypeConstant
	 * @generated
	 */
	EClass getSensorTypeConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SensorTypeConstant#getSensorType <em>Sensor Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Sensor Type</em>'.
	 * @see nqc.SensorTypeConstant#getSensorType()
	 * @see #getSensorTypeConstant()
	 * @generated
	 */
	EAttribute getSensorTypeConstant_SensorType();

	/**
	 * Returns the meta object for class '{@link nqc.SerialBaudConstant <em>Serial Baud Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Serial Baud Constant</em>'.
	 * @see nqc.SerialBaudConstant
	 * @generated
	 */
	EClass getSerialBaudConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SerialBaudConstant#getSerialBaud <em>Serial Baud</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Serial Baud</em>'.
	 * @see nqc.SerialBaudConstant#getSerialBaud()
	 * @see #getSerialBaudConstant()
	 * @generated
	 */
	EAttribute getSerialBaudConstant_SerialBaud();

	/**
	 * Returns the meta object for class '{@link nqc.SerialBiphaseConstant <em>Serial Biphase Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Serial Biphase Constant</em>'.
	 * @see nqc.SerialBiphaseConstant
	 * @generated
	 */
	EClass getSerialBiphaseConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SerialBiphaseConstant#getSerialBiphase <em>Serial Biphase</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Serial Biphase</em>'.
	 * @see nqc.SerialBiphaseConstant#getSerialBiphase()
	 * @see #getSerialBiphaseConstant()
	 * @generated
	 */
	EAttribute getSerialBiphaseConstant_SerialBiphase();

	/**
	 * Returns the meta object for class '{@link nqc.SerialChecksumConstant <em>Serial Checksum Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Serial Checksum Constant</em>'.
	 * @see nqc.SerialChecksumConstant
	 * @generated
	 */
	EClass getSerialChecksumConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SerialChecksumConstant#getSerialChecksum <em>Serial Checksum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Serial Checksum</em>'.
	 * @see nqc.SerialChecksumConstant#getSerialChecksum()
	 * @see #getSerialChecksumConstant()
	 * @generated
	 */
	EAttribute getSerialChecksumConstant_SerialChecksum();

	/**
	 * Returns the meta object for class '{@link nqc.SerialChannelConstant <em>Serial Channel Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Serial Channel Constant</em>'.
	 * @see nqc.SerialChannelConstant
	 * @generated
	 */
	EClass getSerialChannelConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SerialChannelConstant#getSerialChannel <em>Serial Channel</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Serial Channel</em>'.
	 * @see nqc.SerialChannelConstant#getSerialChannel()
	 * @see #getSerialChannelConstant()
	 * @generated
	 */
	EAttribute getSerialChannelConstant_SerialChannel();

	/**
	 * Returns the meta object for class '{@link nqc.SerialCommConstant <em>Serial Comm Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Serial Comm Constant</em>'.
	 * @see nqc.SerialCommConstant
	 * @generated
	 */
	EClass getSerialCommConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SerialCommConstant#getSerialComm <em>Serial Comm</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Serial Comm</em>'.
	 * @see nqc.SerialCommConstant#getSerialComm()
	 * @see #getSerialCommConstant()
	 * @generated
	 */
	EAttribute getSerialCommConstant_SerialComm();

	/**
	 * Returns the meta object for class '{@link nqc.SerialPacketConstant <em>Serial Packet Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Serial Packet Constant</em>'.
	 * @see nqc.SerialPacketConstant
	 * @generated
	 */
	EClass getSerialPacketConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SerialPacketConstant#getSerialPacket <em>Serial Packet</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Serial Packet</em>'.
	 * @see nqc.SerialPacketConstant#getSerialPacket()
	 * @see #getSerialPacketConstant()
	 * @generated
	 */
	EAttribute getSerialPacketConstant_SerialPacket();

	/**
	 * Returns the meta object for class '{@link nqc.SoundConstant <em>Sound Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Sound Constant</em>'.
	 * @see nqc.SoundConstant
	 * @generated
	 */
	EClass getSoundConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.SoundConstant#getSound <em>Sound</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Sound</em>'.
	 * @see nqc.SoundConstant#getSound()
	 * @see #getSoundConstant()
	 * @generated
	 */
	EAttribute getSoundConstant_Sound();

	/**
	 * Returns the meta object for class '{@link nqc.Statement <em>Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Statement</em>'.
	 * @see nqc.Statement
	 * @generated
	 */
	EClass getStatement();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.Statement#getLabel <em>Label</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Label</em>'.
	 * @see nqc.Statement#getLabel()
	 * @see #getStatement()
	 * @generated
	 */
	EReference getStatement_Label();

	/**
	 * Returns the meta object for class '{@link nqc.StartStatement <em>Start Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Start Statement</em>'.
	 * @see nqc.StartStatement
	 * @generated
	 */
	EClass getStartStatement();

	/**
	 * Returns the meta object for the reference '{@link nqc.StartStatement#getTask <em>Task</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Task</em>'.
	 * @see nqc.StartStatement#getTask()
	 * @see #getStartStatement()
	 * @generated
	 */
	EReference getStartStatement_Task();

	/**
	 * Returns the meta object for class '{@link nqc.StopStatement <em>Stop Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Stop Statement</em>'.
	 * @see nqc.StopStatement
	 * @generated
	 */
	EClass getStopStatement();

	/**
	 * Returns the meta object for the reference '{@link nqc.StopStatement#getTask <em>Task</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Task</em>'.
	 * @see nqc.StopStatement#getTask()
	 * @see #getStopStatement()
	 * @generated
	 */
	EReference getStopStatement_Task();

	/**
	 * Returns the meta object for class '{@link nqc.Subroutine <em>Subroutine</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Subroutine</em>'.
	 * @see nqc.Subroutine
	 * @generated
	 */
	EClass getSubroutine();

	/**
	 * Returns the meta object for the attribute '{@link nqc.Subroutine#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see nqc.Subroutine#getName()
	 * @see #getSubroutine()
	 * @generated
	 */
	EAttribute getSubroutine_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Subroutine#getStatements <em>Statements</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Statements</em>'.
	 * @see nqc.Subroutine#getStatements()
	 * @see #getSubroutine()
	 * @generated
	 */
	EReference getSubroutine_Statements();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Subroutine#getLocalVariables <em>Local Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Local Variables</em>'.
	 * @see nqc.Subroutine#getLocalVariables()
	 * @see #getSubroutine()
	 * @generated
	 */
	EReference getSubroutine_LocalVariables();

	/**
	 * Returns the meta object for class '{@link nqc.SubroutineCall <em>Subroutine Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Subroutine Call</em>'.
	 * @see nqc.SubroutineCall
	 * @generated
	 */
	EClass getSubroutineCall();

	/**
	 * Returns the meta object for the reference '{@link nqc.SubroutineCall#getCallee <em>Callee</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Callee</em>'.
	 * @see nqc.SubroutineCall#getCallee()
	 * @see #getSubroutineCall()
	 * @generated
	 */
	EReference getSubroutineCall_Callee();

	/**
	 * Returns the meta object for class '{@link nqc.SwitchStatement <em>Switch Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Switch Statement</em>'.
	 * @see nqc.SwitchStatement
	 * @generated
	 */
	EClass getSwitchStatement();

	/**
	 * Returns the meta object for the reference '{@link nqc.SwitchStatement#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Condition</em>'.
	 * @see nqc.SwitchStatement#getCondition()
	 * @see #getSwitchStatement()
	 * @generated
	 */
	EReference getSwitchStatement_Condition();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.SwitchStatement#getCases <em>Cases</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Cases</em>'.
	 * @see nqc.SwitchStatement#getCases()
	 * @see #getSwitchStatement()
	 * @generated
	 */
	EReference getSwitchStatement_Cases();

	/**
	 * Returns the meta object for class '{@link nqc.Task <em>Task</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Task</em>'.
	 * @see nqc.Task
	 * @generated
	 */
	EClass getTask();

	/**
	 * Returns the meta object for the attribute '{@link nqc.Task#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see nqc.Task#getName()
	 * @see #getTask()
	 * @generated
	 */
	EAttribute getTask_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Task#getStatements <em>Statements</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Statements</em>'.
	 * @see nqc.Task#getStatements()
	 * @see #getTask()
	 * @generated
	 */
	EReference getTask_Statements();

	/**
	 * Returns the meta object for the containment reference list '{@link nqc.Task#getLocalVariables <em>Local Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Local Variables</em>'.
	 * @see nqc.Task#getLocalVariables()
	 * @see #getTask()
	 * @generated
	 */
	EReference getTask_LocalVariables();

	/**
	 * Returns the meta object for class '{@link nqc.TernaryBuiltInFunctionCall <em>Ternary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Ternary Built In Function Call</em>'.
	 * @see nqc.TernaryBuiltInFunctionCall
	 * @generated
	 */
	EClass getTernaryBuiltInFunctionCall();

	/**
	 * Returns the meta object for the attribute '{@link nqc.TernaryBuiltInFunctionCall#getTernaryBuiltInFunction <em>Ternary Built In Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Ternary Built In Function</em>'.
	 * @see nqc.TernaryBuiltInFunctionCall#getTernaryBuiltInFunction()
	 * @see #getTernaryBuiltInFunctionCall()
	 * @generated
	 */
	EAttribute getTernaryBuiltInFunctionCall_TernaryBuiltInFunction();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.TernaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter1</em>'.
	 * @see nqc.TernaryBuiltInFunctionCall#getParameter1()
	 * @see #getTernaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getTernaryBuiltInFunctionCall_Parameter1();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.TernaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter2</em>'.
	 * @see nqc.TernaryBuiltInFunctionCall#getParameter2()
	 * @see #getTernaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getTernaryBuiltInFunctionCall_Parameter2();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.TernaryBuiltInFunctionCall#getParameter3 <em>Parameter3</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter3</em>'.
	 * @see nqc.TernaryBuiltInFunctionCall#getParameter3()
	 * @see #getTernaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getTernaryBuiltInFunctionCall_Parameter3();

	/**
	 * Returns the meta object for class '{@link nqc.TernaryExpression <em>Ternary Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Ternary Expression</em>'.
	 * @see nqc.TernaryExpression
	 * @generated
	 */
	EClass getTernaryExpression();

	/**
	 * Returns the meta object for the attribute '{@link nqc.TernaryExpression#getOperator <em>Operator</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Operator</em>'.
	 * @see nqc.TernaryExpression#getOperator()
	 * @see #getTernaryExpression()
	 * @generated
	 */
	EAttribute getTernaryExpression_Operator();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.TernaryExpression#getOperand1 <em>Operand1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand1</em>'.
	 * @see nqc.TernaryExpression#getOperand1()
	 * @see #getTernaryExpression()
	 * @generated
	 */
	EReference getTernaryExpression_Operand1();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.TernaryExpression#getOperand2 <em>Operand2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand2</em>'.
	 * @see nqc.TernaryExpression#getOperand2()
	 * @see #getTernaryExpression()
	 * @generated
	 */
	EReference getTernaryExpression_Operand2();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.TernaryExpression#getOperand3 <em>Operand3</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand3</em>'.
	 * @see nqc.TernaryExpression#getOperand3()
	 * @see #getTernaryExpression()
	 * @generated
	 */
	EReference getTernaryExpression_Operand3();

	/**
	 * Returns the meta object for class '{@link nqc.TxPowerConstant <em>Tx Power Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Tx Power Constant</em>'.
	 * @see nqc.TxPowerConstant
	 * @generated
	 */
	EClass getTxPowerConstant();

	/**
	 * Returns the meta object for the attribute '{@link nqc.TxPowerConstant#getTxPower <em>Tx Power</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Tx Power</em>'.
	 * @see nqc.TxPowerConstant#getTxPower()
	 * @see #getTxPowerConstant()
	 * @generated
	 */
	EAttribute getTxPowerConstant_TxPower();

	/**
	 * Returns the meta object for class '{@link nqc.UnaryBuiltInFunctionCall <em>Unary Built In Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Unary Built In Function Call</em>'.
	 * @see nqc.UnaryBuiltInFunctionCall
	 * @generated
	 */
	EClass getUnaryBuiltInFunctionCall();

	/**
	 * Returns the meta object for the attribute '{@link nqc.UnaryBuiltInFunctionCall#getUnaryBuiltInFunction <em>Unary Built In Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Unary Built In Function</em>'.
	 * @see nqc.UnaryBuiltInFunctionCall#getUnaryBuiltInFunction()
	 * @see #getUnaryBuiltInFunctionCall()
	 * @generated
	 */
	EAttribute getUnaryBuiltInFunctionCall_UnaryBuiltInFunction();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.UnaryBuiltInFunctionCall#getParameter <em>Parameter</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter</em>'.
	 * @see nqc.UnaryBuiltInFunctionCall#getParameter()
	 * @see #getUnaryBuiltInFunctionCall()
	 * @generated
	 */
	EReference getUnaryBuiltInFunctionCall_Parameter();

	/**
	 * Returns the meta object for class '{@link nqc.UnaryBuiltInValueFunctionCall <em>Unary Built In Value Function Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Unary Built In Value Function Call</em>'.
	 * @see nqc.UnaryBuiltInValueFunctionCall
	 * @generated
	 */
	EClass getUnaryBuiltInValueFunctionCall();

	/**
	 * Returns the meta object for the attribute '{@link nqc.UnaryBuiltInValueFunctionCall#getUnaryBuiltInValueFunction <em>Unary Built In Value Function</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Unary Built In Value Function</em>'.
	 * @see nqc.UnaryBuiltInValueFunctionCall#getUnaryBuiltInValueFunction()
	 * @see #getUnaryBuiltInValueFunctionCall()
	 * @generated
	 */
	EAttribute getUnaryBuiltInValueFunctionCall_UnaryBuiltInValueFunction();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.UnaryBuiltInValueFunctionCall#getParameter <em>Parameter</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Parameter</em>'.
	 * @see nqc.UnaryBuiltInValueFunctionCall#getParameter()
	 * @see #getUnaryBuiltInValueFunctionCall()
	 * @generated
	 */
	EReference getUnaryBuiltInValueFunctionCall_Parameter();

	/**
	 * Returns the meta object for class '{@link nqc.UnaryExpression <em>Unary Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Unary Expression</em>'.
	 * @see nqc.UnaryExpression
	 * @generated
	 */
	EClass getUnaryExpression();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.UnaryExpression#getOperand <em>Operand</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand</em>'.
	 * @see nqc.UnaryExpression#getOperand()
	 * @see #getUnaryExpression()
	 * @generated
	 */
	EReference getUnaryExpression_Operand();

	/**
	 * Returns the meta object for the attribute '{@link nqc.UnaryExpression#getOperator <em>Operator</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Operator</em>'.
	 * @see nqc.UnaryExpression#getOperator()
	 * @see #getUnaryExpression()
	 * @generated
	 */
	EAttribute getUnaryExpression_Operator();

	/**
	 * Returns the meta object for class '{@link nqc.UntilStatement <em>Until Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Until Statement</em>'.
	 * @see nqc.UntilStatement
	 * @generated
	 */
	EClass getUntilStatement();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.UntilStatement#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Condition</em>'.
	 * @see nqc.UntilStatement#getCondition()
	 * @see #getUntilStatement()
	 * @generated
	 */
	EReference getUntilStatement_Condition();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.UntilStatement#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Body</em>'.
	 * @see nqc.UntilStatement#getBody()
	 * @see #getUntilStatement()
	 * @generated
	 */
	EReference getUntilStatement_Body();

	/**
	 * Returns the meta object for class '{@link nqc.ValueExpression <em>Value Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Value Expression</em>'.
	 * @see nqc.ValueExpression
	 * @generated
	 */
	EClass getValueExpression();

	/**
	 * Returns the meta object for class '{@link nqc.Variable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Variable</em>'.
	 * @see nqc.Variable
	 * @generated
	 */
	EClass getVariable();

	/**
	 * Returns the meta object for the attribute '{@link nqc.Variable#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see nqc.Variable#getName()
	 * @see #getVariable()
	 * @generated
	 */
	EAttribute getVariable_Name();

	/**
	 * Returns the meta object for the attribute '{@link nqc.Variable#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Type</em>'.
	 * @see nqc.Variable#getType()
	 * @see #getVariable()
	 * @generated
	 */
	EAttribute getVariable_Type();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.Variable#getInitialValue <em>Initial Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Initial Value</em>'.
	 * @see nqc.Variable#getInitialValue()
	 * @see #getVariable()
	 * @generated
	 */
	EReference getVariable_InitialValue();

	/**
	 * Returns the meta object for the reference '{@link nqc.Variable#getArrayMaxSize <em>Array Max Size</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Array Max Size</em>'.
	 * @see nqc.Variable#getArrayMaxSize()
	 * @see #getVariable()
	 * @generated
	 */
	EReference getVariable_ArrayMaxSize();

	/**
	 * Returns the meta object for class '{@link nqc.VariableExpression <em>Variable Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Variable Expression</em>'.
	 * @see nqc.VariableExpression
	 * @generated
	 */
	EClass getVariableExpression();

	/**
	 * Returns the meta object for the reference '{@link nqc.VariableExpression#getVariable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Variable</em>'.
	 * @see nqc.VariableExpression#getVariable()
	 * @see #getVariableExpression()
	 * @generated
	 */
	EReference getVariableExpression_Variable();

	/**
	 * Returns the meta object for class '{@link nqc.WhileStatement <em>While Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>While Statement</em>'.
	 * @see nqc.WhileStatement
	 * @generated
	 */
	EClass getWhileStatement();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.WhileStatement#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Condition</em>'.
	 * @see nqc.WhileStatement#getCondition()
	 * @see #getWhileStatement()
	 * @generated
	 */
	EReference getWhileStatement_Condition();

	/**
	 * Returns the meta object for the containment reference '{@link nqc.WhileStatement#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Body</em>'.
	 * @see nqc.WhileStatement#getBody()
	 * @see #getWhileStatement()
	 * @generated
	 */
	EReference getWhileStatement_Body();

	/**
	 * Returns the meta object for enum '{@link nqc.AcquireEnum <em>Acquire Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Acquire Enum</em>'.
	 * @see nqc.AcquireEnum
	 * @generated
	 */
	EEnum getAcquireEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.AssignmentStatementEnum <em>Assignment Statement Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Assignment Statement Enum</em>'.
	 * @see nqc.AssignmentStatementEnum
	 * @generated
	 */
	EEnum getAssignmentStatementEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.BinaryOperatorEnum <em>Binary Operator Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Binary Operator Enum</em>'.
	 * @see nqc.BinaryOperatorEnum
	 * @generated
	 */
	EEnum getBinaryOperatorEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.BuiltInBinaryFunctionEnum <em>Built In Binary Function Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Built In Binary Function Enum</em>'.
	 * @see nqc.BuiltInBinaryFunctionEnum
	 * @generated
	 */
	EEnum getBuiltInBinaryFunctionEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.BuiltInBinaryValueFunctionEnum <em>Built In Binary Value Function Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Built In Binary Value Function Enum</em>'.
	 * @see nqc.BuiltInBinaryValueFunctionEnum
	 * @generated
	 */
	EEnum getBuiltInBinaryValueFunctionEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.BuiltInNullaryFunctionEnum <em>Built In Nullary Function Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Built In Nullary Function Enum</em>'.
	 * @see nqc.BuiltInNullaryFunctionEnum
	 * @generated
	 */
	EEnum getBuiltInNullaryFunctionEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.BuiltInNullaryValueFunctionEnum <em>Built In Nullary Value Function Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Built In Nullary Value Function Enum</em>'.
	 * @see nqc.BuiltInNullaryValueFunctionEnum
	 * @generated
	 */
	EEnum getBuiltInNullaryValueFunctionEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.BuiltInUnaryValueFunctionEnum <em>Built In Unary Value Function Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Built In Unary Value Function Enum</em>'.
	 * @see nqc.BuiltInUnaryValueFunctionEnum
	 * @generated
	 */
	EEnum getBuiltInUnaryValueFunctionEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.BuiltInUnaryFunctionEnum <em>Built In Unary Function Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Built In Unary Function Enum</em>'.
	 * @see nqc.BuiltInUnaryFunctionEnum
	 * @generated
	 */
	EEnum getBuiltInUnaryFunctionEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.BuiltInQuaternaryFunctionEnum <em>Built In Quaternary Function Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Built In Quaternary Function Enum</em>'.
	 * @see nqc.BuiltInQuaternaryFunctionEnum
	 * @generated
	 */
	EEnum getBuiltInQuaternaryFunctionEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.BuiltInSenaryFunctionEnum <em>Built In Senary Function Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Built In Senary Function Enum</em>'.
	 * @see nqc.BuiltInSenaryFunctionEnum
	 * @generated
	 */
	EEnum getBuiltInSenaryFunctionEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.BuiltInTernaryFunctionEnum <em>Built In Ternary Function Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Built In Ternary Function Enum</em>'.
	 * @see nqc.BuiltInTernaryFunctionEnum
	 * @generated
	 */
	EEnum getBuiltInTernaryFunctionEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.DirectionEnum <em>Direction Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Direction Enum</em>'.
	 * @see nqc.DirectionEnum
	 * @generated
	 */
	EEnum getDirectionEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.DisplayModeEnum <em>Display Mode Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Display Mode Enum</em>'.
	 * @see nqc.DisplayModeEnum
	 * @generated
	 */
	EEnum getDisplayModeEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.EventTypeEnum <em>Event Type Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Event Type Enum</em>'.
	 * @see nqc.EventTypeEnum
	 * @generated
	 */
	EEnum getEventTypeEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.OutputModeEnum <em>Output Mode Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Output Mode Enum</em>'.
	 * @see nqc.OutputModeEnum
	 * @generated
	 */
	EEnum getOutputModeEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.OutputPortNameEnum <em>Output Port Name Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Output Port Name Enum</em>'.
	 * @see nqc.OutputPortNameEnum
	 * @generated
	 */
	EEnum getOutputPortNameEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SerialBaudEnum <em>Serial Baud Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Serial Baud Enum</em>'.
	 * @see nqc.SerialBaudEnum
	 * @generated
	 */
	EEnum getSerialBaudEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SerialBiphaseEnum <em>Serial Biphase Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Serial Biphase Enum</em>'.
	 * @see nqc.SerialBiphaseEnum
	 * @generated
	 */
	EEnum getSerialBiphaseEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SerialChannelEnum <em>Serial Channel Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Serial Channel Enum</em>'.
	 * @see nqc.SerialChannelEnum
	 * @generated
	 */
	EEnum getSerialChannelEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SerialChecksumEnum <em>Serial Checksum Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Serial Checksum Enum</em>'.
	 * @see nqc.SerialChecksumEnum
	 * @generated
	 */
	EEnum getSerialChecksumEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SerialCommEnum <em>Serial Comm Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Serial Comm Enum</em>'.
	 * @see nqc.SerialCommEnum
	 * @generated
	 */
	EEnum getSerialCommEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SerialPacketEnum <em>Serial Packet Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Serial Packet Enum</em>'.
	 * @see nqc.SerialPacketEnum
	 * @generated
	 */
	EEnum getSerialPacketEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SensorConfigEnum <em>Sensor Config Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Sensor Config Enum</em>'.
	 * @see nqc.SensorConfigEnum
	 * @generated
	 */
	EEnum getSensorConfigEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SensorModeEnum <em>Sensor Mode Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Sensor Mode Enum</em>'.
	 * @see nqc.SensorModeEnum
	 * @generated
	 */
	EEnum getSensorModeEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SensorNameEnum <em>Sensor Name Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Sensor Name Enum</em>'.
	 * @see nqc.SensorNameEnum
	 * @generated
	 */
	EEnum getSensorNameEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SensorTypeEnum <em>Sensor Type Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Sensor Type Enum</em>'.
	 * @see nqc.SensorTypeEnum
	 * @generated
	 */
	EEnum getSensorTypeEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.SoundEnum <em>Sound Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Sound Enum</em>'.
	 * @see nqc.SoundEnum
	 * @generated
	 */
	EEnum getSoundEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.TernaryOperatorEnum <em>Ternary Operator Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Ternary Operator Enum</em>'.
	 * @see nqc.TernaryOperatorEnum
	 * @generated
	 */
	EEnum getTernaryOperatorEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.TxPowerEnum <em>Tx Power Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Tx Power Enum</em>'.
	 * @see nqc.TxPowerEnum
	 * @generated
	 */
	EEnum getTxPowerEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.TypeEnum <em>Type Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Type Enum</em>'.
	 * @see nqc.TypeEnum
	 * @generated
	 */
	EEnum getTypeEnum();

	/**
	 * Returns the meta object for enum '{@link nqc.UnaryOperatorEnum <em>Unary Operator Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Unary Operator Enum</em>'.
	 * @see nqc.UnaryOperatorEnum
	 * @generated
	 */
	EEnum getUnaryOperatorEnum();

	/**
	 * Returns the factory that creates the instances of the model.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the factory that creates the instances of the model.
	 * @generated
	 */
	NqcFactory getNqcFactory();

} //NqcPackage
