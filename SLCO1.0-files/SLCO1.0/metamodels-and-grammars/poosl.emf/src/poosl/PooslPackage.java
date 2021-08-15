/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;

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
 * @see poosl.PooslFactory
 * @model kind="package"
 * @generated
 */
public interface PooslPackage extends EPackage {
	/**
	 * The package name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "poosl";

	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "poosl";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_PREFIX = "poosl";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	PooslPackage eINSTANCE = poosl.impl.PooslPackageImpl.init();

	/**
	 * The meta object id for the '{@link poosl.impl.StatementImpl <em>Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.StatementImpl
	 * @see poosl.impl.PooslPackageImpl#getStatement()
	 * @generated
	 */
	int STATEMENT = 40;

	/**
	 * The number of structural features of the '<em>Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATEMENT_FEATURE_COUNT = 0;

	/**
	 * The meta object id for the '{@link poosl.impl.AbortImpl <em>Abort</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.AbortImpl
	 * @see poosl.impl.PooslPackageImpl#getAbort()
	 * @generated
	 */
	int ABORT = 0;

	/**
	 * The feature id for the '<em><b>Body</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ABORT__BODY = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ABORT__CONDITION = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Abort</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ABORT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link poosl.impl.ExpressionImpl <em>Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ExpressionImpl
	 * @see poosl.impl.PooslPackageImpl#getExpression()
	 * @generated
	 */
	int EXPRESSION = 15;

	/**
	 * The number of structural features of the '<em>Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EXPRESSION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link poosl.impl.AssignmentImpl <em>Assignment</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.AssignmentImpl
	 * @see poosl.impl.PooslPackageImpl#getAssignment()
	 * @generated
	 */
	int ASSIGNMENT = 1;

	/**
	 * The feature id for the '<em><b>Right Hand Side</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGNMENT__RIGHT_HAND_SIDE = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Left Hand Side</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGNMENT__LEFT_HAND_SIDE = EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Assignment</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGNMENT_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link poosl.impl.AtomicImpl <em>Atomic</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.AtomicImpl
	 * @see poosl.impl.PooslPackageImpl#getAtomic()
	 * @generated
	 */
	int ATOMIC = 2;

	/**
	 * The feature id for the '<em><b>Statements</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ATOMIC__STATEMENTS = 0;

	/**
	 * The number of structural features of the '<em>Atomic</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ATOMIC_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link poosl.impl.ChannelImpl <em>Channel</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ChannelImpl
	 * @see poosl.impl.PooslPackageImpl#getChannel()
	 * @generated
	 */
	int CHANNEL = 3;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL__NAME = 0;

	/**
	 * The feature id for the '<em><b>Port1</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL__PORT1 = 1;

	/**
	 * The feature id for the '<em><b>Port2</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL__PORT2 = 2;

	/**
	 * The feature id for the '<em><b>Process Instance1</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL__PROCESS_INSTANCE1 = 3;

	/**
	 * The feature id for the '<em><b>Process Instance2</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL__PROCESS_INSTANCE2 = 4;

	/**
	 * The number of structural features of the '<em>Channel</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL_FEATURE_COUNT = 5;

	/**
	 * The meta object id for the '{@link poosl.impl.TypeImpl <em>Type</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.TypeImpl
	 * @see poosl.impl.PooslPackageImpl#getType()
	 * @generated
	 */
	int TYPE = 43;

	/**
	 * The number of structural features of the '<em>Type</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TYPE_FEATURE_COUNT = 0;

	/**
	 * The meta object id for the '{@link poosl.impl.ClassImpl <em>Class</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ClassImpl
	 * @see poosl.impl.PooslPackageImpl#getClass_()
	 * @generated
	 */
	int CLASS = 4;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLASS__NAME = TYPE_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Class</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLASS_FEATURE_COUNT = TYPE_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link poosl.impl.ConditionalExpressionImpl <em>Conditional Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ConditionalExpressionImpl
	 * @see poosl.impl.PooslPackageImpl#getConditionalExpression()
	 * @generated
	 */
	int CONDITIONAL_EXPRESSION = 5;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONDITIONAL_EXPRESSION__CONDITION = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Then Clause</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONDITIONAL_EXPRESSION__THEN_CLAUSE = EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Else Clause</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONDITIONAL_EXPRESSION__ELSE_CLAUSE = EXPRESSION_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Conditional Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONDITIONAL_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link poosl.impl.ConditionalStatementImpl <em>Conditional Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ConditionalStatementImpl
	 * @see poosl.impl.PooslPackageImpl#getConditionalStatement()
	 * @generated
	 */
	int CONDITIONAL_STATEMENT = 6;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONDITIONAL_STATEMENT__CONDITION = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Then Clause</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONDITIONAL_STATEMENT__THEN_CLAUSE = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Else Clause</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONDITIONAL_STATEMENT__ELSE_CLAUSE = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Conditional Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONDITIONAL_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link poosl.impl.ConstantImpl <em>Constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ConstantImpl
	 * @see poosl.impl.PooslPackageImpl#getConstant()
	 * @generated
	 */
	int CONSTANT = 7;

	/**
	 * The feature id for the '<em><b>Integer</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT__INTEGER = 0;

	/**
	 * The feature id for the '<em><b>String</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT__STRING = 1;

	/**
	 * The feature id for the '<em><b>Boolean</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT__BOOLEAN = 2;

	/**
	 * The feature id for the '<em><b>Real</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT__REAL = 3;

	/**
	 * The feature id for the '<em><b>Type</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT__TYPE = 4;

	/**
	 * The number of structural features of the '<em>Constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT_FEATURE_COUNT = 5;

	/**
	 * The meta object id for the '{@link poosl.impl.ConstantExpressionImpl <em>Constant Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ConstantExpressionImpl
	 * @see poosl.impl.PooslPackageImpl#getConstantExpression()
	 * @generated
	 */
	int CONSTANT_EXPRESSION = 8;

	/**
	 * The feature id for the '<em><b>Constant</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT_EXPRESSION__CONSTANT = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Constant Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link poosl.impl.CurrentTimeImpl <em>Current Time</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.CurrentTimeImpl
	 * @see poosl.impl.PooslPackageImpl#getCurrentTime()
	 * @generated
	 */
	int CURRENT_TIME = 9;

	/**
	 * The number of structural features of the '<em>Current Time</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CURRENT_TIME_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link poosl.impl.DataClassImpl <em>Data Class</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.DataClassImpl
	 * @see poosl.impl.PooslPackageImpl#getDataClass()
	 * @generated
	 */
	int DATA_CLASS = 10;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_CLASS__NAME = CLASS__NAME;

	/**
	 * The feature id for the '<em><b>Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_CLASS__VARIABLES = CLASS_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Methods</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_CLASS__METHODS = CLASS_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Super Class</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_CLASS__SUPER_CLASS = CLASS_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Data Class</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_CLASS_FEATURE_COUNT = CLASS_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link poosl.impl.DataMethodImpl <em>Data Method</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.DataMethodImpl
	 * @see poosl.impl.PooslPackageImpl#getDataMethod()
	 * @generated
	 */
	int DATA_METHOD = 11;

	/**
	 * The feature id for the '<em><b>Parameters</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD__PARAMETERS = 0;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD__NAME = 1;

	/**
	 * The feature id for the '<em><b>Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD__VARIABLES = 2;

	/**
	 * The feature id for the '<em><b>Body</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD__BODY = 3;

	/**
	 * The feature id for the '<em><b>Return Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD__RETURN_TYPE = 4;

	/**
	 * The number of structural features of the '<em>Data Method</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD_FEATURE_COUNT = 5;

	/**
	 * The meta object id for the '{@link poosl.impl.DataMethodCallImpl <em>Data Method Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.DataMethodCallImpl
	 * @see poosl.impl.PooslPackageImpl#getDataMethodCall()
	 * @generated
	 */
	int DATA_METHOD_CALL = 12;

	/**
	 * The feature id for the '<em><b>Super</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD_CALL__SUPER = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Object Expressions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD_CALL__OBJECT_EXPRESSIONS = EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Parameter Expression Lists</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS = EXPRESSION_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Data Method</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD_CALL__DATA_METHOD = EXPRESSION_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Data Method Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_METHOD_CALL_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 4;

	/**
	 * The meta object id for the '{@link poosl.impl.PrimitiveDataMethodCallImpl <em>Primitive Data Method Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.PrimitiveDataMethodCallImpl
	 * @see poosl.impl.PooslPackageImpl#getPrimitiveDataMethodCall()
	 * @generated
	 */
	int PRIMITIVE_DATA_METHOD_CALL = 13;

	/**
	 * The feature id for the '<em><b>Object Expressions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRIMITIVE_DATA_METHOD_CALL__OBJECT_EXPRESSIONS = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Parameter Expression Lists</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRIMITIVE_DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS = EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Primitive Data Method</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRIMITIVE_DATA_METHOD_CALL__PRIMITIVE_DATA_METHOD = EXPRESSION_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Primitive Data Method Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRIMITIVE_DATA_METHOD_CALL_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link poosl.impl.DataObjectCreationImpl <em>Data Object Creation</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.DataObjectCreationImpl
	 * @see poosl.impl.PooslPackageImpl#getDataObjectCreation()
	 * @generated
	 */
	int DATA_OBJECT_CREATION = 14;

	/**
	 * The feature id for the '<em><b>Data Class</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_OBJECT_CREATION__DATA_CLASS = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Data Object Creation</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DATA_OBJECT_CREATION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link poosl.impl.ExpressionListImpl <em>Expression List</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ExpressionListImpl
	 * @see poosl.impl.PooslPackageImpl#getExpressionList()
	 * @generated
	 */
	int EXPRESSION_LIST = 16;

	/**
	 * The feature id for the '<em><b>Expressions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EXPRESSION_LIST__EXPRESSIONS = 0;

	/**
	 * The number of structural features of the '<em>Expression List</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EXPRESSION_LIST_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link poosl.impl.GuardedExecutionImpl <em>Guarded Execution</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.GuardedExecutionImpl
	 * @see poosl.impl.PooslPackageImpl#getGuardedExecution()
	 * @generated
	 */
	int GUARDED_EXECUTION = 17;

	/**
	 * The feature id for the '<em><b>Statement</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int GUARDED_EXECUTION__STATEMENT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Guard</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int GUARDED_EXECUTION__GUARD = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Guarded Execution</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int GUARDED_EXECUTION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link poosl.impl.MessageImpl <em>Message</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.MessageImpl
	 * @see poosl.impl.PooslPackageImpl#getMessage()
	 * @generated
	 */
	int MESSAGE = 22;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE__NAME = 0;

	/**
	 * The feature id for the '<em><b>Parameter Types</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE__PARAMETER_TYPES = 1;

	/**
	 * The number of structural features of the '<em>Message</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_FEATURE_COUNT = 2;

	/**
	 * The meta object id for the '{@link poosl.impl.IncomingMessageImpl <em>Incoming Message</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.IncomingMessageImpl
	 * @see poosl.impl.PooslPackageImpl#getIncomingMessage()
	 * @generated
	 */
	int INCOMING_MESSAGE = 18;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INCOMING_MESSAGE__NAME = MESSAGE__NAME;

	/**
	 * The feature id for the '<em><b>Parameter Types</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INCOMING_MESSAGE__PARAMETER_TYPES = MESSAGE__PARAMETER_TYPES;

	/**
	 * The number of structural features of the '<em>Incoming Message</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INCOMING_MESSAGE_FEATURE_COUNT = MESSAGE_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link poosl.impl.InterruptImpl <em>Interrupt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.InterruptImpl
	 * @see poosl.impl.PooslPackageImpl#getInterrupt()
	 * @generated
	 */
	int INTERRUPT = 19;

	/**
	 * The feature id for the '<em><b>Body</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERRUPT__BODY = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERRUPT__CONDITION = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Interrupt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERRUPT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link poosl.impl.LoopExpressionImpl <em>Loop Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.LoopExpressionImpl
	 * @see poosl.impl.PooslPackageImpl#getLoopExpression()
	 * @generated
	 */
	int LOOP_EXPRESSION = 20;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LOOP_EXPRESSION__CONDITION = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Body</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LOOP_EXPRESSION__BODY = EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Loop Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LOOP_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link poosl.impl.LoopStatementImpl <em>Loop Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.LoopStatementImpl
	 * @see poosl.impl.PooslPackageImpl#getLoopStatement()
	 * @generated
	 */
	int LOOP_STATEMENT = 21;

	/**
	 * The feature id for the '<em><b>Body</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LOOP_STATEMENT__BODY = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LOOP_STATEMENT__CONDITION = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Loop Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LOOP_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link poosl.impl.MessageReceiveImpl <em>Message Receive</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.MessageReceiveImpl
	 * @see poosl.impl.PooslPackageImpl#getMessageReceive()
	 * @generated
	 */
	int MESSAGE_RECEIVE = 23;

	/**
	 * The feature id for the '<em><b>Message</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_RECEIVE__MESSAGE = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_RECEIVE__PORT = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Variables</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_RECEIVE__VARIABLES = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_RECEIVE__CONDITION = STATEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Expressions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_RECEIVE__EXPRESSIONS = STATEMENT_FEATURE_COUNT + 4;

	/**
	 * The number of structural features of the '<em>Message Receive</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_RECEIVE_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 5;

	/**
	 * The meta object id for the '{@link poosl.impl.MessageSendImpl <em>Message Send</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.MessageSendImpl
	 * @see poosl.impl.PooslPackageImpl#getMessageSend()
	 * @generated
	 */
	int MESSAGE_SEND = 24;

	/**
	 * The feature id for the '<em><b>Message</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_SEND__MESSAGE = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_SEND__PORT = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Expressions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_SEND__EXPRESSIONS = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Expression Lists</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_SEND__EXPRESSION_LISTS = STATEMENT_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Message Send</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MESSAGE_SEND_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 4;

	/**
	 * The meta object id for the '{@link poosl.impl.ModelImpl <em>Model</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ModelImpl
	 * @see poosl.impl.PooslPackageImpl#getModel()
	 * @generated
	 */
	int MODEL = 25;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MODEL__NAME = 0;

	/**
	 * The feature id for the '<em><b>Classes</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MODEL__CLASSES = 1;

	/**
	 * The feature id for the '<em><b>Process Instances</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MODEL__PROCESS_INSTANCES = 2;

	/**
	 * The feature id for the '<em><b>Channels</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MODEL__CHANNELS = 3;

	/**
	 * The feature id for the '<em><b>Primitive Types</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MODEL__PRIMITIVE_TYPES = 4;

	/**
	 * The number of structural features of the '<em>Model</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MODEL_FEATURE_COUNT = 5;

	/**
	 * The meta object id for the '{@link poosl.impl.OutgoingMessageImpl <em>Outgoing Message</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.OutgoingMessageImpl
	 * @see poosl.impl.PooslPackageImpl#getOutgoingMessage()
	 * @generated
	 */
	int OUTGOING_MESSAGE = 26;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OUTGOING_MESSAGE__NAME = MESSAGE__NAME;

	/**
	 * The feature id for the '<em><b>Parameter Types</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OUTGOING_MESSAGE__PARAMETER_TYPES = MESSAGE__PARAMETER_TYPES;

	/**
	 * The number of structural features of the '<em>Outgoing Message</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OUTGOING_MESSAGE_FEATURE_COUNT = MESSAGE_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link poosl.impl.ParallelCompositionImpl <em>Parallel Composition</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ParallelCompositionImpl
	 * @see poosl.impl.PooslPackageImpl#getParallelComposition()
	 * @generated
	 */
	int PARALLEL_COMPOSITION = 27;

	/**
	 * The feature id for the '<em><b>Statement Lists</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARALLEL_COMPOSITION__STATEMENT_LISTS = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Parallel Composition</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARALLEL_COMPOSITION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link poosl.impl.ParameterImpl <em>Parameter</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ParameterImpl
	 * @see poosl.impl.PooslPackageImpl#getParameter()
	 * @generated
	 */
	int PARAMETER = 28;

	/**
	 * The feature id for the '<em><b>Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__TYPE = 0;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__NAME = 1;

	/**
	 * The number of structural features of the '<em>Parameter</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_FEATURE_COUNT = 2;

	/**
	 * The meta object id for the '{@link poosl.impl.ParameterExpressionImpl <em>Parameter Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ParameterExpressionImpl
	 * @see poosl.impl.PooslPackageImpl#getParameterExpression()
	 * @generated
	 */
	int PARAMETER_EXPRESSION = 29;

	/**
	 * The feature id for the '<em><b>Parameter</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_EXPRESSION__PARAMETER = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Parameter Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link poosl.impl.PrimitiveTypeImpl <em>Primitive Type</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.PrimitiveTypeImpl
	 * @see poosl.impl.PooslPackageImpl#getPrimitiveType()
	 * @generated
	 */
	int PRIMITIVE_TYPE = 30;

	/**
	 * The feature id for the '<em><b>Primitive Type Enum</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRIMITIVE_TYPE__PRIMITIVE_TYPE_ENUM = TYPE_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Primitive Type</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRIMITIVE_TYPE_FEATURE_COUNT = TYPE_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link poosl.impl.ProcessClassImpl <em>Process Class</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ProcessClassImpl
	 * @see poosl.impl.PooslPackageImpl#getProcessClass()
	 * @generated
	 */
	int PROCESS_CLASS = 31;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_CLASS__NAME = CLASS__NAME;

	/**
	 * The feature id for the '<em><b>Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_CLASS__VARIABLES = CLASS_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Methods</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_CLASS__METHODS = CLASS_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Super Class</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_CLASS__SUPER_CLASS = CLASS_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Parameters</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_CLASS__PARAMETERS = CLASS_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Ports</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_CLASS__PORTS = CLASS_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Initial Method Call</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_CLASS__INITIAL_METHOD_CALL = CLASS_FEATURE_COUNT + 5;

	/**
	 * The number of structural features of the '<em>Process Class</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_CLASS_FEATURE_COUNT = CLASS_FEATURE_COUNT + 6;

	/**
	 * The meta object id for the '{@link poosl.impl.ProcessInstanceImpl <em>Process Instance</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ProcessInstanceImpl
	 * @see poosl.impl.PooslPackageImpl#getProcessInstance()
	 * @generated
	 */
	int PROCESS_INSTANCE = 32;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_INSTANCE__NAME = 0;

	/**
	 * The feature id for the '<em><b>Class</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_INSTANCE__CLASS = 1;

	/**
	 * The number of structural features of the '<em>Process Instance</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_INSTANCE_FEATURE_COUNT = 2;

	/**
	 * The meta object id for the '{@link poosl.impl.ProcessMethodImpl <em>Process Method</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ProcessMethodImpl
	 * @see poosl.impl.PooslPackageImpl#getProcessMethod()
	 * @generated
	 */
	int PROCESS_METHOD = 33;

	/**
	 * The feature id for the '<em><b>Input Parameters</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_METHOD__INPUT_PARAMETERS = 0;

	/**
	 * The feature id for the '<em><b>Output Parameters</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_METHOD__OUTPUT_PARAMETERS = 1;

	/**
	 * The feature id for the '<em><b>Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_METHOD__VARIABLES = 2;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_METHOD__NAME = 3;

	/**
	 * The feature id for the '<em><b>Body</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_METHOD__BODY = 4;

	/**
	 * The number of structural features of the '<em>Process Method</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_METHOD_FEATURE_COUNT = 5;

	/**
	 * The meta object id for the '{@link poosl.impl.ProcessMethodCallImpl <em>Process Method Call</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ProcessMethodCallImpl
	 * @see poosl.impl.PooslPackageImpl#getProcessMethodCall()
	 * @generated
	 */
	int PROCESS_METHOD_CALL = 34;

	/**
	 * The feature id for the '<em><b>Method</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_METHOD_CALL__METHOD = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Parameter Expressions</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_METHOD_CALL__PARAMETER_EXPRESSIONS = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Variables</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_METHOD_CALL__VARIABLES = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Process Method Call</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCESS_METHOD_CALL_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link poosl.impl.PortImpl <em>Port</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.PortImpl
	 * @see poosl.impl.PooslPackageImpl#getPort()
	 * @generated
	 */
	int PORT = 35;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PORT__NAME = 0;

	/**
	 * The number of structural features of the '<em>Port</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PORT_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link poosl.impl.ReferenceSelfImpl <em>Reference Self</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ReferenceSelfImpl
	 * @see poosl.impl.PooslPackageImpl#getReferenceSelf()
	 * @generated
	 */
	int REFERENCE_SELF = 36;

	/**
	 * The number of structural features of the '<em>Reference Self</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REFERENCE_SELF_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link poosl.impl.ReturnExpressionImpl <em>Return Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.ReturnExpressionImpl
	 * @see poosl.impl.PooslPackageImpl#getReturnExpression()
	 * @generated
	 */
	int RETURN_EXPRESSION = 37;

	/**
	 * The feature id for the '<em><b>Return Expressions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RETURN_EXPRESSION__RETURN_EXPRESSIONS = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Return Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RETURN_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link poosl.impl.SelectionImpl <em>Selection</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.SelectionImpl
	 * @see poosl.impl.PooslPackageImpl#getSelection()
	 * @generated
	 */
	int SELECTION = 38;

	/**
	 * The feature id for the '<em><b>Statement Lists</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SELECTION__STATEMENT_LISTS = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Selection</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SELECTION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link poosl.impl.SkipImpl <em>Skip</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.SkipImpl
	 * @see poosl.impl.PooslPackageImpl#getSkip()
	 * @generated
	 */
	int SKIP = 39;

	/**
	 * The number of structural features of the '<em>Skip</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SKIP_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link poosl.impl.StatementListImpl <em>Statement List</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.StatementListImpl
	 * @see poosl.impl.PooslPackageImpl#getStatementList()
	 * @generated
	 */
	int STATEMENT_LIST = 41;

	/**
	 * The feature id for the '<em><b>Statements</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATEMENT_LIST__STATEMENTS = 0;

	/**
	 * The number of structural features of the '<em>Statement List</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATEMENT_LIST_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link poosl.impl.TimeSynchronisationImpl <em>Time Synchronisation</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.TimeSynchronisationImpl
	 * @see poosl.impl.PooslPackageImpl#getTimeSynchronisation()
	 * @generated
	 */
	int TIME_SYNCHRONISATION = 42;

	/**
	 * The feature id for the '<em><b>Duration</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TIME_SYNCHRONISATION__DURATION = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Time Synchronisation</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TIME_SYNCHRONISATION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link poosl.impl.VariableImpl <em>Variable</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.VariableImpl
	 * @see poosl.impl.PooslPackageImpl#getVariable()
	 * @generated
	 */
	int VARIABLE = 44;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE__NAME = 0;

	/**
	 * The feature id for the '<em><b>Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE__TYPE = 1;

	/**
	 * The number of structural features of the '<em>Variable</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE_FEATURE_COUNT = 2;

	/**
	 * The meta object id for the '{@link poosl.impl.VariableExpressionImpl <em>Variable Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.impl.VariableExpressionImpl
	 * @see poosl.impl.PooslPackageImpl#getVariableExpression()
	 * @generated
	 */
	int VARIABLE_EXPRESSION = 45;

	/**
	 * The feature id for the '<em><b>Variable</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE_EXPRESSION__VARIABLE = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Variable Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link poosl.PrimitiveTypeEnum <em>Primitive Type Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.PrimitiveTypeEnum
	 * @see poosl.impl.PooslPackageImpl#getPrimitiveTypeEnum()
	 * @generated
	 */
	int PRIMITIVE_TYPE_ENUM = 46;

	/**
	 * The meta object id for the '{@link poosl.PrimitiveDataMethodEnum <em>Primitive Data Method Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see poosl.PrimitiveDataMethodEnum
	 * @see poosl.impl.PooslPackageImpl#getPrimitiveDataMethodEnum()
	 * @generated
	 */
	int PRIMITIVE_DATA_METHOD_ENUM = 47;


	/**
	 * Returns the meta object for class '{@link poosl.Abort <em>Abort</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Abort</em>'.
	 * @see poosl.Abort
	 * @generated
	 */
	EClass getAbort();

	/**
	 * Returns the meta object for the reference list '{@link poosl.Abort#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Body</em>'.
	 * @see poosl.Abort#getBody()
	 * @see #getAbort()
	 * @generated
	 */
	EReference getAbort_Body();

	/**
	 * Returns the meta object for the reference list '{@link poosl.Abort#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Condition</em>'.
	 * @see poosl.Abort#getCondition()
	 * @see #getAbort()
	 * @generated
	 */
	EReference getAbort_Condition();

	/**
	 * Returns the meta object for class '{@link poosl.Assignment <em>Assignment</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Assignment</em>'.
	 * @see poosl.Assignment
	 * @generated
	 */
	EClass getAssignment();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.Assignment#getRightHandSide <em>Right Hand Side</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Right Hand Side</em>'.
	 * @see poosl.Assignment#getRightHandSide()
	 * @see #getAssignment()
	 * @generated
	 */
	EReference getAssignment_RightHandSide();

	/**
	 * Returns the meta object for the reference '{@link poosl.Assignment#getLeftHandSide <em>Left Hand Side</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Left Hand Side</em>'.
	 * @see poosl.Assignment#getLeftHandSide()
	 * @see #getAssignment()
	 * @generated
	 */
	EReference getAssignment_LeftHandSide();

	/**
	 * Returns the meta object for class '{@link poosl.Atomic <em>Atomic</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Atomic</em>'.
	 * @see poosl.Atomic
	 * @generated
	 */
	EClass getAtomic();

	/**
	 * Returns the meta object for the reference list '{@link poosl.Atomic#getStatements <em>Statements</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Statements</em>'.
	 * @see poosl.Atomic#getStatements()
	 * @see #getAtomic()
	 * @generated
	 */
	EReference getAtomic_Statements();

	/**
	 * Returns the meta object for class '{@link poosl.Channel <em>Channel</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Channel</em>'.
	 * @see poosl.Channel
	 * @generated
	 */
	EClass getChannel();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Channel#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see poosl.Channel#getName()
	 * @see #getChannel()
	 * @generated
	 */
	EAttribute getChannel_Name();

	/**
	 * Returns the meta object for the reference '{@link poosl.Channel#getPort1 <em>Port1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port1</em>'.
	 * @see poosl.Channel#getPort1()
	 * @see #getChannel()
	 * @generated
	 */
	EReference getChannel_Port1();

	/**
	 * Returns the meta object for the reference '{@link poosl.Channel#getPort2 <em>Port2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port2</em>'.
	 * @see poosl.Channel#getPort2()
	 * @see #getChannel()
	 * @generated
	 */
	EReference getChannel_Port2();

	/**
	 * Returns the meta object for the reference '{@link poosl.Channel#getProcessInstance1 <em>Process Instance1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Process Instance1</em>'.
	 * @see poosl.Channel#getProcessInstance1()
	 * @see #getChannel()
	 * @generated
	 */
	EReference getChannel_ProcessInstance1();

	/**
	 * Returns the meta object for the reference '{@link poosl.Channel#getProcessInstance2 <em>Process Instance2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Process Instance2</em>'.
	 * @see poosl.Channel#getProcessInstance2()
	 * @see #getChannel()
	 * @generated
	 */
	EReference getChannel_ProcessInstance2();

	/**
	 * Returns the meta object for class '{@link poosl.Class <em>Class</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Class</em>'.
	 * @see poosl.Class
	 * @generated
	 */
	EClass getClass_();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Class#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see poosl.Class#getName()
	 * @see #getClass_()
	 * @generated
	 */
	EAttribute getClass_Name();

	/**
	 * Returns the meta object for class '{@link poosl.ConditionalExpression <em>Conditional Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Conditional Expression</em>'.
	 * @see poosl.ConditionalExpression
	 * @generated
	 */
	EClass getConditionalExpression();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ConditionalExpression#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Condition</em>'.
	 * @see poosl.ConditionalExpression#getCondition()
	 * @see #getConditionalExpression()
	 * @generated
	 */
	EReference getConditionalExpression_Condition();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ConditionalExpression#getThenClause <em>Then Clause</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Then Clause</em>'.
	 * @see poosl.ConditionalExpression#getThenClause()
	 * @see #getConditionalExpression()
	 * @generated
	 */
	EReference getConditionalExpression_ThenClause();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ConditionalExpression#getElseClause <em>Else Clause</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Else Clause</em>'.
	 * @see poosl.ConditionalExpression#getElseClause()
	 * @see #getConditionalExpression()
	 * @generated
	 */
	EReference getConditionalExpression_ElseClause();

	/**
	 * Returns the meta object for class '{@link poosl.ConditionalStatement <em>Conditional Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Conditional Statement</em>'.
	 * @see poosl.ConditionalStatement
	 * @generated
	 */
	EClass getConditionalStatement();

	/**
	 * Returns the meta object for the reference list '{@link poosl.ConditionalStatement#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Condition</em>'.
	 * @see poosl.ConditionalStatement#getCondition()
	 * @see #getConditionalStatement()
	 * @generated
	 */
	EReference getConditionalStatement_Condition();

	/**
	 * Returns the meta object for the reference list '{@link poosl.ConditionalStatement#getThenClause <em>Then Clause</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Then Clause</em>'.
	 * @see poosl.ConditionalStatement#getThenClause()
	 * @see #getConditionalStatement()
	 * @generated
	 */
	EReference getConditionalStatement_ThenClause();

	/**
	 * Returns the meta object for the reference list '{@link poosl.ConditionalStatement#getElseClause <em>Else Clause</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Else Clause</em>'.
	 * @see poosl.ConditionalStatement#getElseClause()
	 * @see #getConditionalStatement()
	 * @generated
	 */
	EReference getConditionalStatement_ElseClause();

	/**
	 * Returns the meta object for class '{@link poosl.Constant <em>Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Constant</em>'.
	 * @see poosl.Constant
	 * @generated
	 */
	EClass getConstant();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Constant#getInteger <em>Integer</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Integer</em>'.
	 * @see poosl.Constant#getInteger()
	 * @see #getConstant()
	 * @generated
	 */
	EAttribute getConstant_Integer();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Constant#getString <em>String</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>String</em>'.
	 * @see poosl.Constant#getString()
	 * @see #getConstant()
	 * @generated
	 */
	EAttribute getConstant_String();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Constant#isBoolean <em>Boolean</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Boolean</em>'.
	 * @see poosl.Constant#isBoolean()
	 * @see #getConstant()
	 * @generated
	 */
	EAttribute getConstant_Boolean();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Constant#getReal <em>Real</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Real</em>'.
	 * @see poosl.Constant#getReal()
	 * @see #getConstant()
	 * @generated
	 */
	EAttribute getConstant_Real();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Constant#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Type</em>'.
	 * @see poosl.Constant#getType()
	 * @see #getConstant()
	 * @generated
	 */
	EAttribute getConstant_Type();

	/**
	 * Returns the meta object for class '{@link poosl.ConstantExpression <em>Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Constant Expression</em>'.
	 * @see poosl.ConstantExpression
	 * @generated
	 */
	EClass getConstantExpression();

	/**
	 * Returns the meta object for the containment reference '{@link poosl.ConstantExpression#getConstant <em>Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Constant</em>'.
	 * @see poosl.ConstantExpression#getConstant()
	 * @see #getConstantExpression()
	 * @generated
	 */
	EReference getConstantExpression_Constant();

	/**
	 * Returns the meta object for class '{@link poosl.CurrentTime <em>Current Time</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Current Time</em>'.
	 * @see poosl.CurrentTime
	 * @generated
	 */
	EClass getCurrentTime();

	/**
	 * Returns the meta object for class '{@link poosl.DataClass <em>Data Class</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Data Class</em>'.
	 * @see poosl.DataClass
	 * @generated
	 */
	EClass getDataClass();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.DataClass#getVariables <em>Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Variables</em>'.
	 * @see poosl.DataClass#getVariables()
	 * @see #getDataClass()
	 * @generated
	 */
	EReference getDataClass_Variables();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.DataClass#getMethods <em>Methods</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Methods</em>'.
	 * @see poosl.DataClass#getMethods()
	 * @see #getDataClass()
	 * @generated
	 */
	EReference getDataClass_Methods();

	/**
	 * Returns the meta object for the reference '{@link poosl.DataClass#getSuperClass <em>Super Class</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Super Class</em>'.
	 * @see poosl.DataClass#getSuperClass()
	 * @see #getDataClass()
	 * @generated
	 */
	EReference getDataClass_SuperClass();

	/**
	 * Returns the meta object for class '{@link poosl.DataMethod <em>Data Method</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Data Method</em>'.
	 * @see poosl.DataMethod
	 * @generated
	 */
	EClass getDataMethod();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.DataMethod#getParameters <em>Parameters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Parameters</em>'.
	 * @see poosl.DataMethod#getParameters()
	 * @see #getDataMethod()
	 * @generated
	 */
	EReference getDataMethod_Parameters();

	/**
	 * Returns the meta object for the attribute '{@link poosl.DataMethod#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see poosl.DataMethod#getName()
	 * @see #getDataMethod()
	 * @generated
	 */
	EAttribute getDataMethod_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.DataMethod#getVariables <em>Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Variables</em>'.
	 * @see poosl.DataMethod#getVariables()
	 * @see #getDataMethod()
	 * @generated
	 */
	EReference getDataMethod_Variables();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.DataMethod#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Body</em>'.
	 * @see poosl.DataMethod#getBody()
	 * @see #getDataMethod()
	 * @generated
	 */
	EReference getDataMethod_Body();

	/**
	 * Returns the meta object for the reference '{@link poosl.DataMethod#getReturnType <em>Return Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Return Type</em>'.
	 * @see poosl.DataMethod#getReturnType()
	 * @see #getDataMethod()
	 * @generated
	 */
	EReference getDataMethod_ReturnType();

	/**
	 * Returns the meta object for class '{@link poosl.DataMethodCall <em>Data Method Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Data Method Call</em>'.
	 * @see poosl.DataMethodCall
	 * @generated
	 */
	EClass getDataMethodCall();

	/**
	 * Returns the meta object for the attribute '{@link poosl.DataMethodCall#isSuper <em>Super</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Super</em>'.
	 * @see poosl.DataMethodCall#isSuper()
	 * @see #getDataMethodCall()
	 * @generated
	 */
	EAttribute getDataMethodCall_Super();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.DataMethodCall#getObjectExpressions <em>Object Expressions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Object Expressions</em>'.
	 * @see poosl.DataMethodCall#getObjectExpressions()
	 * @see #getDataMethodCall()
	 * @generated
	 */
	EReference getDataMethodCall_ObjectExpressions();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.DataMethodCall#getParameterExpressionLists <em>Parameter Expression Lists</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Parameter Expression Lists</em>'.
	 * @see poosl.DataMethodCall#getParameterExpressionLists()
	 * @see #getDataMethodCall()
	 * @generated
	 */
	EReference getDataMethodCall_ParameterExpressionLists();

	/**
	 * Returns the meta object for the reference '{@link poosl.DataMethodCall#getDataMethod <em>Data Method</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Data Method</em>'.
	 * @see poosl.DataMethodCall#getDataMethod()
	 * @see #getDataMethodCall()
	 * @generated
	 */
	EReference getDataMethodCall_DataMethod();

	/**
	 * Returns the meta object for class '{@link poosl.PrimitiveDataMethodCall <em>Primitive Data Method Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Primitive Data Method Call</em>'.
	 * @see poosl.PrimitiveDataMethodCall
	 * @generated
	 */
	EClass getPrimitiveDataMethodCall();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.PrimitiveDataMethodCall#getObjectExpressions <em>Object Expressions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Object Expressions</em>'.
	 * @see poosl.PrimitiveDataMethodCall#getObjectExpressions()
	 * @see #getPrimitiveDataMethodCall()
	 * @generated
	 */
	EReference getPrimitiveDataMethodCall_ObjectExpressions();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.PrimitiveDataMethodCall#getParameterExpressionLists <em>Parameter Expression Lists</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Parameter Expression Lists</em>'.
	 * @see poosl.PrimitiveDataMethodCall#getParameterExpressionLists()
	 * @see #getPrimitiveDataMethodCall()
	 * @generated
	 */
	EReference getPrimitiveDataMethodCall_ParameterExpressionLists();

	/**
	 * Returns the meta object for the attribute '{@link poosl.PrimitiveDataMethodCall#getPrimitiveDataMethod <em>Primitive Data Method</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Primitive Data Method</em>'.
	 * @see poosl.PrimitiveDataMethodCall#getPrimitiveDataMethod()
	 * @see #getPrimitiveDataMethodCall()
	 * @generated
	 */
	EAttribute getPrimitiveDataMethodCall_PrimitiveDataMethod();

	/**
	 * Returns the meta object for class '{@link poosl.DataObjectCreation <em>Data Object Creation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Data Object Creation</em>'.
	 * @see poosl.DataObjectCreation
	 * @generated
	 */
	EClass getDataObjectCreation();

	/**
	 * Returns the meta object for the reference '{@link poosl.DataObjectCreation#getDataClass <em>Data Class</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Data Class</em>'.
	 * @see poosl.DataObjectCreation#getDataClass()
	 * @see #getDataObjectCreation()
	 * @generated
	 */
	EReference getDataObjectCreation_DataClass();

	/**
	 * Returns the meta object for class '{@link poosl.Expression <em>Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Expression</em>'.
	 * @see poosl.Expression
	 * @generated
	 */
	EClass getExpression();

	/**
	 * Returns the meta object for class '{@link poosl.ExpressionList <em>Expression List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Expression List</em>'.
	 * @see poosl.ExpressionList
	 * @generated
	 */
	EClass getExpressionList();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ExpressionList#getExpressions <em>Expressions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Expressions</em>'.
	 * @see poosl.ExpressionList#getExpressions()
	 * @see #getExpressionList()
	 * @generated
	 */
	EReference getExpressionList_Expressions();

	/**
	 * Returns the meta object for class '{@link poosl.GuardedExecution <em>Guarded Execution</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Guarded Execution</em>'.
	 * @see poosl.GuardedExecution
	 * @generated
	 */
	EClass getGuardedExecution();

	/**
	 * Returns the meta object for the containment reference '{@link poosl.GuardedExecution#getStatement <em>Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Statement</em>'.
	 * @see poosl.GuardedExecution#getStatement()
	 * @see #getGuardedExecution()
	 * @generated
	 */
	EReference getGuardedExecution_Statement();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.GuardedExecution#getGuard <em>Guard</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Guard</em>'.
	 * @see poosl.GuardedExecution#getGuard()
	 * @see #getGuardedExecution()
	 * @generated
	 */
	EReference getGuardedExecution_Guard();

	/**
	 * Returns the meta object for class '{@link poosl.IncomingMessage <em>Incoming Message</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Incoming Message</em>'.
	 * @see poosl.IncomingMessage
	 * @generated
	 */
	EClass getIncomingMessage();

	/**
	 * Returns the meta object for class '{@link poosl.Interrupt <em>Interrupt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Interrupt</em>'.
	 * @see poosl.Interrupt
	 * @generated
	 */
	EClass getInterrupt();

	/**
	 * Returns the meta object for the reference list '{@link poosl.Interrupt#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Body</em>'.
	 * @see poosl.Interrupt#getBody()
	 * @see #getInterrupt()
	 * @generated
	 */
	EReference getInterrupt_Body();

	/**
	 * Returns the meta object for the reference list '{@link poosl.Interrupt#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Condition</em>'.
	 * @see poosl.Interrupt#getCondition()
	 * @see #getInterrupt()
	 * @generated
	 */
	EReference getInterrupt_Condition();

	/**
	 * Returns the meta object for class '{@link poosl.LoopExpression <em>Loop Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Loop Expression</em>'.
	 * @see poosl.LoopExpression
	 * @generated
	 */
	EClass getLoopExpression();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.LoopExpression#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Condition</em>'.
	 * @see poosl.LoopExpression#getCondition()
	 * @see #getLoopExpression()
	 * @generated
	 */
	EReference getLoopExpression_Condition();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.LoopExpression#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Body</em>'.
	 * @see poosl.LoopExpression#getBody()
	 * @see #getLoopExpression()
	 * @generated
	 */
	EReference getLoopExpression_Body();

	/**
	 * Returns the meta object for class '{@link poosl.LoopStatement <em>Loop Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Loop Statement</em>'.
	 * @see poosl.LoopStatement
	 * @generated
	 */
	EClass getLoopStatement();

	/**
	 * Returns the meta object for the reference list '{@link poosl.LoopStatement#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Body</em>'.
	 * @see poosl.LoopStatement#getBody()
	 * @see #getLoopStatement()
	 * @generated
	 */
	EReference getLoopStatement_Body();

	/**
	 * Returns the meta object for the reference list '{@link poosl.LoopStatement#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Condition</em>'.
	 * @see poosl.LoopStatement#getCondition()
	 * @see #getLoopStatement()
	 * @generated
	 */
	EReference getLoopStatement_Condition();

	/**
	 * Returns the meta object for class '{@link poosl.Message <em>Message</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Message</em>'.
	 * @see poosl.Message
	 * @generated
	 */
	EClass getMessage();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Message#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see poosl.Message#getName()
	 * @see #getMessage()
	 * @generated
	 */
	EAttribute getMessage_Name();

	/**
	 * Returns the meta object for the reference list '{@link poosl.Message#getParameterTypes <em>Parameter Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Parameter Types</em>'.
	 * @see poosl.Message#getParameterTypes()
	 * @see #getMessage()
	 * @generated
	 */
	EReference getMessage_ParameterTypes();

	/**
	 * Returns the meta object for class '{@link poosl.MessageReceive <em>Message Receive</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Message Receive</em>'.
	 * @see poosl.MessageReceive
	 * @generated
	 */
	EClass getMessageReceive();

	/**
	 * Returns the meta object for the containment reference '{@link poosl.MessageReceive#getMessage <em>Message</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Message</em>'.
	 * @see poosl.MessageReceive#getMessage()
	 * @see #getMessageReceive()
	 * @generated
	 */
	EReference getMessageReceive_Message();

	/**
	 * Returns the meta object for the reference '{@link poosl.MessageReceive#getPort <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port</em>'.
	 * @see poosl.MessageReceive#getPort()
	 * @see #getMessageReceive()
	 * @generated
	 */
	EReference getMessageReceive_Port();

	/**
	 * Returns the meta object for the reference list '{@link poosl.MessageReceive#getVariables <em>Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Variables</em>'.
	 * @see poosl.MessageReceive#getVariables()
	 * @see #getMessageReceive()
	 * @generated
	 */
	EReference getMessageReceive_Variables();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.MessageReceive#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Condition</em>'.
	 * @see poosl.MessageReceive#getCondition()
	 * @see #getMessageReceive()
	 * @generated
	 */
	EReference getMessageReceive_Condition();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.MessageReceive#getExpressions <em>Expressions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Expressions</em>'.
	 * @see poosl.MessageReceive#getExpressions()
	 * @see #getMessageReceive()
	 * @generated
	 */
	EReference getMessageReceive_Expressions();

	/**
	 * Returns the meta object for class '{@link poosl.MessageSend <em>Message Send</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Message Send</em>'.
	 * @see poosl.MessageSend
	 * @generated
	 */
	EClass getMessageSend();

	/**
	 * Returns the meta object for the containment reference '{@link poosl.MessageSend#getMessage <em>Message</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Message</em>'.
	 * @see poosl.MessageSend#getMessage()
	 * @see #getMessageSend()
	 * @generated
	 */
	EReference getMessageSend_Message();

	/**
	 * Returns the meta object for the reference '{@link poosl.MessageSend#getPort <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port</em>'.
	 * @see poosl.MessageSend#getPort()
	 * @see #getMessageSend()
	 * @generated
	 */
	EReference getMessageSend_Port();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.MessageSend#getExpressions <em>Expressions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Expressions</em>'.
	 * @see poosl.MessageSend#getExpressions()
	 * @see #getMessageSend()
	 * @generated
	 */
	EReference getMessageSend_Expressions();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.MessageSend#getExpressionLists <em>Expression Lists</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Expression Lists</em>'.
	 * @see poosl.MessageSend#getExpressionLists()
	 * @see #getMessageSend()
	 * @generated
	 */
	EReference getMessageSend_ExpressionLists();

	/**
	 * Returns the meta object for class '{@link poosl.Model <em>Model</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Model</em>'.
	 * @see poosl.Model
	 * @generated
	 */
	EClass getModel();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Model#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see poosl.Model#getName()
	 * @see #getModel()
	 * @generated
	 */
	EAttribute getModel_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.Model#getClasses <em>Classes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Classes</em>'.
	 * @see poosl.Model#getClasses()
	 * @see #getModel()
	 * @generated
	 */
	EReference getModel_Classes();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.Model#getProcessInstances <em>Process Instances</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Process Instances</em>'.
	 * @see poosl.Model#getProcessInstances()
	 * @see #getModel()
	 * @generated
	 */
	EReference getModel_ProcessInstances();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.Model#getChannels <em>Channels</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Channels</em>'.
	 * @see poosl.Model#getChannels()
	 * @see #getModel()
	 * @generated
	 */
	EReference getModel_Channels();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.Model#getPrimitiveTypes <em>Primitive Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Primitive Types</em>'.
	 * @see poosl.Model#getPrimitiveTypes()
	 * @see #getModel()
	 * @generated
	 */
	EReference getModel_PrimitiveTypes();

	/**
	 * Returns the meta object for class '{@link poosl.OutgoingMessage <em>Outgoing Message</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Outgoing Message</em>'.
	 * @see poosl.OutgoingMessage
	 * @generated
	 */
	EClass getOutgoingMessage();

	/**
	 * Returns the meta object for class '{@link poosl.ParallelComposition <em>Parallel Composition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Parallel Composition</em>'.
	 * @see poosl.ParallelComposition
	 * @generated
	 */
	EClass getParallelComposition();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ParallelComposition#getStatementLists <em>Statement Lists</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Statement Lists</em>'.
	 * @see poosl.ParallelComposition#getStatementLists()
	 * @see #getParallelComposition()
	 * @generated
	 */
	EReference getParallelComposition_StatementLists();

	/**
	 * Returns the meta object for class '{@link poosl.Parameter <em>Parameter</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Parameter</em>'.
	 * @see poosl.Parameter
	 * @generated
	 */
	EClass getParameter();

	/**
	 * Returns the meta object for the reference '{@link poosl.Parameter#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Type</em>'.
	 * @see poosl.Parameter#getType()
	 * @see #getParameter()
	 * @generated
	 */
	EReference getParameter_Type();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Parameter#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see poosl.Parameter#getName()
	 * @see #getParameter()
	 * @generated
	 */
	EAttribute getParameter_Name();

	/**
	 * Returns the meta object for class '{@link poosl.ParameterExpression <em>Parameter Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Parameter Expression</em>'.
	 * @see poosl.ParameterExpression
	 * @generated
	 */
	EClass getParameterExpression();

	/**
	 * Returns the meta object for the reference '{@link poosl.ParameterExpression#getParameter <em>Parameter</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Parameter</em>'.
	 * @see poosl.ParameterExpression#getParameter()
	 * @see #getParameterExpression()
	 * @generated
	 */
	EReference getParameterExpression_Parameter();

	/**
	 * Returns the meta object for class '{@link poosl.PrimitiveType <em>Primitive Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Primitive Type</em>'.
	 * @see poosl.PrimitiveType
	 * @generated
	 */
	EClass getPrimitiveType();

	/**
	 * Returns the meta object for the attribute '{@link poosl.PrimitiveType#getPrimitiveTypeEnum <em>Primitive Type Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Primitive Type Enum</em>'.
	 * @see poosl.PrimitiveType#getPrimitiveTypeEnum()
	 * @see #getPrimitiveType()
	 * @generated
	 */
	EAttribute getPrimitiveType_PrimitiveTypeEnum();

	/**
	 * Returns the meta object for class '{@link poosl.ProcessClass <em>Process Class</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Process Class</em>'.
	 * @see poosl.ProcessClass
	 * @generated
	 */
	EClass getProcessClass();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ProcessClass#getVariables <em>Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Variables</em>'.
	 * @see poosl.ProcessClass#getVariables()
	 * @see #getProcessClass()
	 * @generated
	 */
	EReference getProcessClass_Variables();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ProcessClass#getMethods <em>Methods</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Methods</em>'.
	 * @see poosl.ProcessClass#getMethods()
	 * @see #getProcessClass()
	 * @generated
	 */
	EReference getProcessClass_Methods();

	/**
	 * Returns the meta object for the reference '{@link poosl.ProcessClass#getSuperClass <em>Super Class</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Super Class</em>'.
	 * @see poosl.ProcessClass#getSuperClass()
	 * @see #getProcessClass()
	 * @generated
	 */
	EReference getProcessClass_SuperClass();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ProcessClass#getParameters <em>Parameters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Parameters</em>'.
	 * @see poosl.ProcessClass#getParameters()
	 * @see #getProcessClass()
	 * @generated
	 */
	EReference getProcessClass_Parameters();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ProcessClass#getPorts <em>Ports</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Ports</em>'.
	 * @see poosl.ProcessClass#getPorts()
	 * @see #getProcessClass()
	 * @generated
	 */
	EReference getProcessClass_Ports();

	/**
	 * Returns the meta object for the containment reference '{@link poosl.ProcessClass#getInitialMethodCall <em>Initial Method Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Initial Method Call</em>'.
	 * @see poosl.ProcessClass#getInitialMethodCall()
	 * @see #getProcessClass()
	 * @generated
	 */
	EReference getProcessClass_InitialMethodCall();

	/**
	 * Returns the meta object for class '{@link poosl.ProcessInstance <em>Process Instance</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Process Instance</em>'.
	 * @see poosl.ProcessInstance
	 * @generated
	 */
	EClass getProcessInstance();

	/**
	 * Returns the meta object for the attribute '{@link poosl.ProcessInstance#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see poosl.ProcessInstance#getName()
	 * @see #getProcessInstance()
	 * @generated
	 */
	EAttribute getProcessInstance_Name();

	/**
	 * Returns the meta object for the reference '{@link poosl.ProcessInstance#getClass_ <em>Class</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Class</em>'.
	 * @see poosl.ProcessInstance#getClass_()
	 * @see #getProcessInstance()
	 * @generated
	 */
	EReference getProcessInstance_Class();

	/**
	 * Returns the meta object for class '{@link poosl.ProcessMethod <em>Process Method</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Process Method</em>'.
	 * @see poosl.ProcessMethod
	 * @generated
	 */
	EClass getProcessMethod();

	/**
	 * Returns the meta object for the reference list '{@link poosl.ProcessMethod#getInputParameters <em>Input Parameters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Input Parameters</em>'.
	 * @see poosl.ProcessMethod#getInputParameters()
	 * @see #getProcessMethod()
	 * @generated
	 */
	EReference getProcessMethod_InputParameters();

	/**
	 * Returns the meta object for the reference list '{@link poosl.ProcessMethod#getOutputParameters <em>Output Parameters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Output Parameters</em>'.
	 * @see poosl.ProcessMethod#getOutputParameters()
	 * @see #getProcessMethod()
	 * @generated
	 */
	EReference getProcessMethod_OutputParameters();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ProcessMethod#getVariables <em>Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Variables</em>'.
	 * @see poosl.ProcessMethod#getVariables()
	 * @see #getProcessMethod()
	 * @generated
	 */
	EReference getProcessMethod_Variables();

	/**
	 * Returns the meta object for the attribute '{@link poosl.ProcessMethod#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see poosl.ProcessMethod#getName()
	 * @see #getProcessMethod()
	 * @generated
	 */
	EAttribute getProcessMethod_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ProcessMethod#getBody <em>Body</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Body</em>'.
	 * @see poosl.ProcessMethod#getBody()
	 * @see #getProcessMethod()
	 * @generated
	 */
	EReference getProcessMethod_Body();

	/**
	 * Returns the meta object for class '{@link poosl.ProcessMethodCall <em>Process Method Call</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Process Method Call</em>'.
	 * @see poosl.ProcessMethodCall
	 * @generated
	 */
	EClass getProcessMethodCall();

	/**
	 * Returns the meta object for the reference '{@link poosl.ProcessMethodCall#getMethod <em>Method</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Method</em>'.
	 * @see poosl.ProcessMethodCall#getMethod()
	 * @see #getProcessMethodCall()
	 * @generated
	 */
	EReference getProcessMethodCall_Method();

	/**
	 * Returns the meta object for the reference list '{@link poosl.ProcessMethodCall#getParameterExpressions <em>Parameter Expressions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Parameter Expressions</em>'.
	 * @see poosl.ProcessMethodCall#getParameterExpressions()
	 * @see #getProcessMethodCall()
	 * @generated
	 */
	EReference getProcessMethodCall_ParameterExpressions();

	/**
	 * Returns the meta object for the reference list '{@link poosl.ProcessMethodCall#getVariables <em>Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Variables</em>'.
	 * @see poosl.ProcessMethodCall#getVariables()
	 * @see #getProcessMethodCall()
	 * @generated
	 */
	EReference getProcessMethodCall_Variables();

	/**
	 * Returns the meta object for class '{@link poosl.Port <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Port</em>'.
	 * @see poosl.Port
	 * @generated
	 */
	EClass getPort();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Port#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see poosl.Port#getName()
	 * @see #getPort()
	 * @generated
	 */
	EAttribute getPort_Name();

	/**
	 * Returns the meta object for class '{@link poosl.ReferenceSelf <em>Reference Self</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Reference Self</em>'.
	 * @see poosl.ReferenceSelf
	 * @generated
	 */
	EClass getReferenceSelf();

	/**
	 * Returns the meta object for class '{@link poosl.ReturnExpression <em>Return Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Return Expression</em>'.
	 * @see poosl.ReturnExpression
	 * @generated
	 */
	EClass getReturnExpression();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.ReturnExpression#getReturnExpressions <em>Return Expressions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Return Expressions</em>'.
	 * @see poosl.ReturnExpression#getReturnExpressions()
	 * @see #getReturnExpression()
	 * @generated
	 */
	EReference getReturnExpression_ReturnExpressions();

	/**
	 * Returns the meta object for class '{@link poosl.Selection <em>Selection</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Selection</em>'.
	 * @see poosl.Selection
	 * @generated
	 */
	EClass getSelection();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.Selection#getStatementLists <em>Statement Lists</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Statement Lists</em>'.
	 * @see poosl.Selection#getStatementLists()
	 * @see #getSelection()
	 * @generated
	 */
	EReference getSelection_StatementLists();

	/**
	 * Returns the meta object for class '{@link poosl.Skip <em>Skip</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Skip</em>'.
	 * @see poosl.Skip
	 * @generated
	 */
	EClass getSkip();

	/**
	 * Returns the meta object for class '{@link poosl.Statement <em>Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Statement</em>'.
	 * @see poosl.Statement
	 * @generated
	 */
	EClass getStatement();

	/**
	 * Returns the meta object for class '{@link poosl.StatementList <em>Statement List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Statement List</em>'.
	 * @see poosl.StatementList
	 * @generated
	 */
	EClass getStatementList();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.StatementList#getStatements <em>Statements</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Statements</em>'.
	 * @see poosl.StatementList#getStatements()
	 * @see #getStatementList()
	 * @generated
	 */
	EReference getStatementList_Statements();

	/**
	 * Returns the meta object for class '{@link poosl.TimeSynchronisation <em>Time Synchronisation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Time Synchronisation</em>'.
	 * @see poosl.TimeSynchronisation
	 * @generated
	 */
	EClass getTimeSynchronisation();

	/**
	 * Returns the meta object for the containment reference list '{@link poosl.TimeSynchronisation#getDuration <em>Duration</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Duration</em>'.
	 * @see poosl.TimeSynchronisation#getDuration()
	 * @see #getTimeSynchronisation()
	 * @generated
	 */
	EReference getTimeSynchronisation_Duration();

	/**
	 * Returns the meta object for class '{@link poosl.Type <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Type</em>'.
	 * @see poosl.Type
	 * @generated
	 */
	EClass getType();

	/**
	 * Returns the meta object for class '{@link poosl.Variable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Variable</em>'.
	 * @see poosl.Variable
	 * @generated
	 */
	EClass getVariable();

	/**
	 * Returns the meta object for the attribute '{@link poosl.Variable#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see poosl.Variable#getName()
	 * @see #getVariable()
	 * @generated
	 */
	EAttribute getVariable_Name();

	/**
	 * Returns the meta object for the reference '{@link poosl.Variable#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Type</em>'.
	 * @see poosl.Variable#getType()
	 * @see #getVariable()
	 * @generated
	 */
	EReference getVariable_Type();

	/**
	 * Returns the meta object for class '{@link poosl.VariableExpression <em>Variable Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Variable Expression</em>'.
	 * @see poosl.VariableExpression
	 * @generated
	 */
	EClass getVariableExpression();

	/**
	 * Returns the meta object for the reference '{@link poosl.VariableExpression#getVariable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Variable</em>'.
	 * @see poosl.VariableExpression#getVariable()
	 * @see #getVariableExpression()
	 * @generated
	 */
	EReference getVariableExpression_Variable();

	/**
	 * Returns the meta object for enum '{@link poosl.PrimitiveTypeEnum <em>Primitive Type Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Primitive Type Enum</em>'.
	 * @see poosl.PrimitiveTypeEnum
	 * @generated
	 */
	EEnum getPrimitiveTypeEnum();

	/**
	 * Returns the meta object for enum '{@link poosl.PrimitiveDataMethodEnum <em>Primitive Data Method Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Primitive Data Method Enum</em>'.
	 * @see poosl.PrimitiveDataMethodEnum
	 * @generated
	 */
	EEnum getPrimitiveDataMethodEnum();

	/**
	 * Returns the factory that creates the instances of the model.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the factory that creates the instances of the model.
	 * @generated
	 */
	PooslFactory getPooslFactory();

	/**
	 * <!-- begin-user-doc -->
	 * Defines literals for the meta objects that represent
	 * <ul>
	 *   <li>each class,</li>
	 *   <li>each feature of each class,</li>
	 *   <li>each enum,</li>
	 *   <li>and each data type</li>
	 * </ul>
	 * <!-- end-user-doc -->
	 * @generated
	 */
	interface Literals {
		/**
		 * The meta object literal for the '{@link poosl.impl.AbortImpl <em>Abort</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.AbortImpl
		 * @see poosl.impl.PooslPackageImpl#getAbort()
		 * @generated
		 */
		EClass ABORT = eINSTANCE.getAbort();

		/**
		 * The meta object literal for the '<em><b>Body</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ABORT__BODY = eINSTANCE.getAbort_Body();

		/**
		 * The meta object literal for the '<em><b>Condition</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ABORT__CONDITION = eINSTANCE.getAbort_Condition();

		/**
		 * The meta object literal for the '{@link poosl.impl.AssignmentImpl <em>Assignment</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.AssignmentImpl
		 * @see poosl.impl.PooslPackageImpl#getAssignment()
		 * @generated
		 */
		EClass ASSIGNMENT = eINSTANCE.getAssignment();

		/**
		 * The meta object literal for the '<em><b>Right Hand Side</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ASSIGNMENT__RIGHT_HAND_SIDE = eINSTANCE.getAssignment_RightHandSide();

		/**
		 * The meta object literal for the '<em><b>Left Hand Side</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ASSIGNMENT__LEFT_HAND_SIDE = eINSTANCE.getAssignment_LeftHandSide();

		/**
		 * The meta object literal for the '{@link poosl.impl.AtomicImpl <em>Atomic</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.AtomicImpl
		 * @see poosl.impl.PooslPackageImpl#getAtomic()
		 * @generated
		 */
		EClass ATOMIC = eINSTANCE.getAtomic();

		/**
		 * The meta object literal for the '<em><b>Statements</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ATOMIC__STATEMENTS = eINSTANCE.getAtomic_Statements();

		/**
		 * The meta object literal for the '{@link poosl.impl.ChannelImpl <em>Channel</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ChannelImpl
		 * @see poosl.impl.PooslPackageImpl#getChannel()
		 * @generated
		 */
		EClass CHANNEL = eINSTANCE.getChannel();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CHANNEL__NAME = eINSTANCE.getChannel_Name();

		/**
		 * The meta object literal for the '<em><b>Port1</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CHANNEL__PORT1 = eINSTANCE.getChannel_Port1();

		/**
		 * The meta object literal for the '<em><b>Port2</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CHANNEL__PORT2 = eINSTANCE.getChannel_Port2();

		/**
		 * The meta object literal for the '<em><b>Process Instance1</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CHANNEL__PROCESS_INSTANCE1 = eINSTANCE.getChannel_ProcessInstance1();

		/**
		 * The meta object literal for the '<em><b>Process Instance2</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CHANNEL__PROCESS_INSTANCE2 = eINSTANCE.getChannel_ProcessInstance2();

		/**
		 * The meta object literal for the '{@link poosl.impl.ClassImpl <em>Class</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ClassImpl
		 * @see poosl.impl.PooslPackageImpl#getClass_()
		 * @generated
		 */
		EClass CLASS = eINSTANCE.getClass_();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CLASS__NAME = eINSTANCE.getClass_Name();

		/**
		 * The meta object literal for the '{@link poosl.impl.ConditionalExpressionImpl <em>Conditional Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ConditionalExpressionImpl
		 * @see poosl.impl.PooslPackageImpl#getConditionalExpression()
		 * @generated
		 */
		EClass CONDITIONAL_EXPRESSION = eINSTANCE.getConditionalExpression();

		/**
		 * The meta object literal for the '<em><b>Condition</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CONDITIONAL_EXPRESSION__CONDITION = eINSTANCE.getConditionalExpression_Condition();

		/**
		 * The meta object literal for the '<em><b>Then Clause</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CONDITIONAL_EXPRESSION__THEN_CLAUSE = eINSTANCE.getConditionalExpression_ThenClause();

		/**
		 * The meta object literal for the '<em><b>Else Clause</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CONDITIONAL_EXPRESSION__ELSE_CLAUSE = eINSTANCE.getConditionalExpression_ElseClause();

		/**
		 * The meta object literal for the '{@link poosl.impl.ConditionalStatementImpl <em>Conditional Statement</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ConditionalStatementImpl
		 * @see poosl.impl.PooslPackageImpl#getConditionalStatement()
		 * @generated
		 */
		EClass CONDITIONAL_STATEMENT = eINSTANCE.getConditionalStatement();

		/**
		 * The meta object literal for the '<em><b>Condition</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CONDITIONAL_STATEMENT__CONDITION = eINSTANCE.getConditionalStatement_Condition();

		/**
		 * The meta object literal for the '<em><b>Then Clause</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CONDITIONAL_STATEMENT__THEN_CLAUSE = eINSTANCE.getConditionalStatement_ThenClause();

		/**
		 * The meta object literal for the '<em><b>Else Clause</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CONDITIONAL_STATEMENT__ELSE_CLAUSE = eINSTANCE.getConditionalStatement_ElseClause();

		/**
		 * The meta object literal for the '{@link poosl.impl.ConstantImpl <em>Constant</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ConstantImpl
		 * @see poosl.impl.PooslPackageImpl#getConstant()
		 * @generated
		 */
		EClass CONSTANT = eINSTANCE.getConstant();

		/**
		 * The meta object literal for the '<em><b>Integer</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CONSTANT__INTEGER = eINSTANCE.getConstant_Integer();

		/**
		 * The meta object literal for the '<em><b>String</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CONSTANT__STRING = eINSTANCE.getConstant_String();

		/**
		 * The meta object literal for the '<em><b>Boolean</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CONSTANT__BOOLEAN = eINSTANCE.getConstant_Boolean();

		/**
		 * The meta object literal for the '<em><b>Real</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CONSTANT__REAL = eINSTANCE.getConstant_Real();

		/**
		 * The meta object literal for the '<em><b>Type</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CONSTANT__TYPE = eINSTANCE.getConstant_Type();

		/**
		 * The meta object literal for the '{@link poosl.impl.ConstantExpressionImpl <em>Constant Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ConstantExpressionImpl
		 * @see poosl.impl.PooslPackageImpl#getConstantExpression()
		 * @generated
		 */
		EClass CONSTANT_EXPRESSION = eINSTANCE.getConstantExpression();

		/**
		 * The meta object literal for the '<em><b>Constant</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CONSTANT_EXPRESSION__CONSTANT = eINSTANCE.getConstantExpression_Constant();

		/**
		 * The meta object literal for the '{@link poosl.impl.CurrentTimeImpl <em>Current Time</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.CurrentTimeImpl
		 * @see poosl.impl.PooslPackageImpl#getCurrentTime()
		 * @generated
		 */
		EClass CURRENT_TIME = eINSTANCE.getCurrentTime();

		/**
		 * The meta object literal for the '{@link poosl.impl.DataClassImpl <em>Data Class</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.DataClassImpl
		 * @see poosl.impl.PooslPackageImpl#getDataClass()
		 * @generated
		 */
		EClass DATA_CLASS = eINSTANCE.getDataClass();

		/**
		 * The meta object literal for the '<em><b>Variables</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_CLASS__VARIABLES = eINSTANCE.getDataClass_Variables();

		/**
		 * The meta object literal for the '<em><b>Methods</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_CLASS__METHODS = eINSTANCE.getDataClass_Methods();

		/**
		 * The meta object literal for the '<em><b>Super Class</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_CLASS__SUPER_CLASS = eINSTANCE.getDataClass_SuperClass();

		/**
		 * The meta object literal for the '{@link poosl.impl.DataMethodImpl <em>Data Method</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.DataMethodImpl
		 * @see poosl.impl.PooslPackageImpl#getDataMethod()
		 * @generated
		 */
		EClass DATA_METHOD = eINSTANCE.getDataMethod();

		/**
		 * The meta object literal for the '<em><b>Parameters</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_METHOD__PARAMETERS = eINSTANCE.getDataMethod_Parameters();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATA_METHOD__NAME = eINSTANCE.getDataMethod_Name();

		/**
		 * The meta object literal for the '<em><b>Variables</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_METHOD__VARIABLES = eINSTANCE.getDataMethod_Variables();

		/**
		 * The meta object literal for the '<em><b>Body</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_METHOD__BODY = eINSTANCE.getDataMethod_Body();

		/**
		 * The meta object literal for the '<em><b>Return Type</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_METHOD__RETURN_TYPE = eINSTANCE.getDataMethod_ReturnType();

		/**
		 * The meta object literal for the '{@link poosl.impl.DataMethodCallImpl <em>Data Method Call</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.DataMethodCallImpl
		 * @see poosl.impl.PooslPackageImpl#getDataMethodCall()
		 * @generated
		 */
		EClass DATA_METHOD_CALL = eINSTANCE.getDataMethodCall();

		/**
		 * The meta object literal for the '<em><b>Super</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute DATA_METHOD_CALL__SUPER = eINSTANCE.getDataMethodCall_Super();

		/**
		 * The meta object literal for the '<em><b>Object Expressions</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_METHOD_CALL__OBJECT_EXPRESSIONS = eINSTANCE.getDataMethodCall_ObjectExpressions();

		/**
		 * The meta object literal for the '<em><b>Parameter Expression Lists</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS = eINSTANCE.getDataMethodCall_ParameterExpressionLists();

		/**
		 * The meta object literal for the '<em><b>Data Method</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_METHOD_CALL__DATA_METHOD = eINSTANCE.getDataMethodCall_DataMethod();

		/**
		 * The meta object literal for the '{@link poosl.impl.PrimitiveDataMethodCallImpl <em>Primitive Data Method Call</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.PrimitiveDataMethodCallImpl
		 * @see poosl.impl.PooslPackageImpl#getPrimitiveDataMethodCall()
		 * @generated
		 */
		EClass PRIMITIVE_DATA_METHOD_CALL = eINSTANCE.getPrimitiveDataMethodCall();

		/**
		 * The meta object literal for the '<em><b>Object Expressions</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PRIMITIVE_DATA_METHOD_CALL__OBJECT_EXPRESSIONS = eINSTANCE.getPrimitiveDataMethodCall_ObjectExpressions();

		/**
		 * The meta object literal for the '<em><b>Parameter Expression Lists</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PRIMITIVE_DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS = eINSTANCE.getPrimitiveDataMethodCall_ParameterExpressionLists();

		/**
		 * The meta object literal for the '<em><b>Primitive Data Method</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PRIMITIVE_DATA_METHOD_CALL__PRIMITIVE_DATA_METHOD = eINSTANCE.getPrimitiveDataMethodCall_PrimitiveDataMethod();

		/**
		 * The meta object literal for the '{@link poosl.impl.DataObjectCreationImpl <em>Data Object Creation</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.DataObjectCreationImpl
		 * @see poosl.impl.PooslPackageImpl#getDataObjectCreation()
		 * @generated
		 */
		EClass DATA_OBJECT_CREATION = eINSTANCE.getDataObjectCreation();

		/**
		 * The meta object literal for the '<em><b>Data Class</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DATA_OBJECT_CREATION__DATA_CLASS = eINSTANCE.getDataObjectCreation_DataClass();

		/**
		 * The meta object literal for the '{@link poosl.impl.ExpressionImpl <em>Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ExpressionImpl
		 * @see poosl.impl.PooslPackageImpl#getExpression()
		 * @generated
		 */
		EClass EXPRESSION = eINSTANCE.getExpression();

		/**
		 * The meta object literal for the '{@link poosl.impl.ExpressionListImpl <em>Expression List</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ExpressionListImpl
		 * @see poosl.impl.PooslPackageImpl#getExpressionList()
		 * @generated
		 */
		EClass EXPRESSION_LIST = eINSTANCE.getExpressionList();

		/**
		 * The meta object literal for the '<em><b>Expressions</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference EXPRESSION_LIST__EXPRESSIONS = eINSTANCE.getExpressionList_Expressions();

		/**
		 * The meta object literal for the '{@link poosl.impl.GuardedExecutionImpl <em>Guarded Execution</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.GuardedExecutionImpl
		 * @see poosl.impl.PooslPackageImpl#getGuardedExecution()
		 * @generated
		 */
		EClass GUARDED_EXECUTION = eINSTANCE.getGuardedExecution();

		/**
		 * The meta object literal for the '<em><b>Statement</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference GUARDED_EXECUTION__STATEMENT = eINSTANCE.getGuardedExecution_Statement();

		/**
		 * The meta object literal for the '<em><b>Guard</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference GUARDED_EXECUTION__GUARD = eINSTANCE.getGuardedExecution_Guard();

		/**
		 * The meta object literal for the '{@link poosl.impl.IncomingMessageImpl <em>Incoming Message</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.IncomingMessageImpl
		 * @see poosl.impl.PooslPackageImpl#getIncomingMessage()
		 * @generated
		 */
		EClass INCOMING_MESSAGE = eINSTANCE.getIncomingMessage();

		/**
		 * The meta object literal for the '{@link poosl.impl.InterruptImpl <em>Interrupt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.InterruptImpl
		 * @see poosl.impl.PooslPackageImpl#getInterrupt()
		 * @generated
		 */
		EClass INTERRUPT = eINSTANCE.getInterrupt();

		/**
		 * The meta object literal for the '<em><b>Body</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference INTERRUPT__BODY = eINSTANCE.getInterrupt_Body();

		/**
		 * The meta object literal for the '<em><b>Condition</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference INTERRUPT__CONDITION = eINSTANCE.getInterrupt_Condition();

		/**
		 * The meta object literal for the '{@link poosl.impl.LoopExpressionImpl <em>Loop Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.LoopExpressionImpl
		 * @see poosl.impl.PooslPackageImpl#getLoopExpression()
		 * @generated
		 */
		EClass LOOP_EXPRESSION = eINSTANCE.getLoopExpression();

		/**
		 * The meta object literal for the '<em><b>Condition</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference LOOP_EXPRESSION__CONDITION = eINSTANCE.getLoopExpression_Condition();

		/**
		 * The meta object literal for the '<em><b>Body</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference LOOP_EXPRESSION__BODY = eINSTANCE.getLoopExpression_Body();

		/**
		 * The meta object literal for the '{@link poosl.impl.LoopStatementImpl <em>Loop Statement</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.LoopStatementImpl
		 * @see poosl.impl.PooslPackageImpl#getLoopStatement()
		 * @generated
		 */
		EClass LOOP_STATEMENT = eINSTANCE.getLoopStatement();

		/**
		 * The meta object literal for the '<em><b>Body</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference LOOP_STATEMENT__BODY = eINSTANCE.getLoopStatement_Body();

		/**
		 * The meta object literal for the '<em><b>Condition</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference LOOP_STATEMENT__CONDITION = eINSTANCE.getLoopStatement_Condition();

		/**
		 * The meta object literal for the '{@link poosl.impl.MessageImpl <em>Message</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.MessageImpl
		 * @see poosl.impl.PooslPackageImpl#getMessage()
		 * @generated
		 */
		EClass MESSAGE = eINSTANCE.getMessage();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MESSAGE__NAME = eINSTANCE.getMessage_Name();

		/**
		 * The meta object literal for the '<em><b>Parameter Types</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MESSAGE__PARAMETER_TYPES = eINSTANCE.getMessage_ParameterTypes();

		/**
		 * The meta object literal for the '{@link poosl.impl.MessageReceiveImpl <em>Message Receive</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.MessageReceiveImpl
		 * @see poosl.impl.PooslPackageImpl#getMessageReceive()
		 * @generated
		 */
		EClass MESSAGE_RECEIVE = eINSTANCE.getMessageReceive();

		/**
		 * The meta object literal for the '<em><b>Message</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MESSAGE_RECEIVE__MESSAGE = eINSTANCE.getMessageReceive_Message();

		/**
		 * The meta object literal for the '<em><b>Port</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MESSAGE_RECEIVE__PORT = eINSTANCE.getMessageReceive_Port();

		/**
		 * The meta object literal for the '<em><b>Variables</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MESSAGE_RECEIVE__VARIABLES = eINSTANCE.getMessageReceive_Variables();

		/**
		 * The meta object literal for the '<em><b>Condition</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MESSAGE_RECEIVE__CONDITION = eINSTANCE.getMessageReceive_Condition();

		/**
		 * The meta object literal for the '<em><b>Expressions</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MESSAGE_RECEIVE__EXPRESSIONS = eINSTANCE.getMessageReceive_Expressions();

		/**
		 * The meta object literal for the '{@link poosl.impl.MessageSendImpl <em>Message Send</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.MessageSendImpl
		 * @see poosl.impl.PooslPackageImpl#getMessageSend()
		 * @generated
		 */
		EClass MESSAGE_SEND = eINSTANCE.getMessageSend();

		/**
		 * The meta object literal for the '<em><b>Message</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MESSAGE_SEND__MESSAGE = eINSTANCE.getMessageSend_Message();

		/**
		 * The meta object literal for the '<em><b>Port</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MESSAGE_SEND__PORT = eINSTANCE.getMessageSend_Port();

		/**
		 * The meta object literal for the '<em><b>Expressions</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MESSAGE_SEND__EXPRESSIONS = eINSTANCE.getMessageSend_Expressions();

		/**
		 * The meta object literal for the '<em><b>Expression Lists</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MESSAGE_SEND__EXPRESSION_LISTS = eINSTANCE.getMessageSend_ExpressionLists();

		/**
		 * The meta object literal for the '{@link poosl.impl.ModelImpl <em>Model</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ModelImpl
		 * @see poosl.impl.PooslPackageImpl#getModel()
		 * @generated
		 */
		EClass MODEL = eINSTANCE.getModel();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MODEL__NAME = eINSTANCE.getModel_Name();

		/**
		 * The meta object literal for the '<em><b>Classes</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MODEL__CLASSES = eINSTANCE.getModel_Classes();

		/**
		 * The meta object literal for the '<em><b>Process Instances</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MODEL__PROCESS_INSTANCES = eINSTANCE.getModel_ProcessInstances();

		/**
		 * The meta object literal for the '<em><b>Channels</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MODEL__CHANNELS = eINSTANCE.getModel_Channels();

		/**
		 * The meta object literal for the '<em><b>Primitive Types</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MODEL__PRIMITIVE_TYPES = eINSTANCE.getModel_PrimitiveTypes();

		/**
		 * The meta object literal for the '{@link poosl.impl.OutgoingMessageImpl <em>Outgoing Message</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.OutgoingMessageImpl
		 * @see poosl.impl.PooslPackageImpl#getOutgoingMessage()
		 * @generated
		 */
		EClass OUTGOING_MESSAGE = eINSTANCE.getOutgoingMessage();

		/**
		 * The meta object literal for the '{@link poosl.impl.ParallelCompositionImpl <em>Parallel Composition</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ParallelCompositionImpl
		 * @see poosl.impl.PooslPackageImpl#getParallelComposition()
		 * @generated
		 */
		EClass PARALLEL_COMPOSITION = eINSTANCE.getParallelComposition();

		/**
		 * The meta object literal for the '<em><b>Statement Lists</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PARALLEL_COMPOSITION__STATEMENT_LISTS = eINSTANCE.getParallelComposition_StatementLists();

		/**
		 * The meta object literal for the '{@link poosl.impl.ParameterImpl <em>Parameter</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ParameterImpl
		 * @see poosl.impl.PooslPackageImpl#getParameter()
		 * @generated
		 */
		EClass PARAMETER = eINSTANCE.getParameter();

		/**
		 * The meta object literal for the '<em><b>Type</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PARAMETER__TYPE = eINSTANCE.getParameter_Type();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PARAMETER__NAME = eINSTANCE.getParameter_Name();

		/**
		 * The meta object literal for the '{@link poosl.impl.ParameterExpressionImpl <em>Parameter Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ParameterExpressionImpl
		 * @see poosl.impl.PooslPackageImpl#getParameterExpression()
		 * @generated
		 */
		EClass PARAMETER_EXPRESSION = eINSTANCE.getParameterExpression();

		/**
		 * The meta object literal for the '<em><b>Parameter</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PARAMETER_EXPRESSION__PARAMETER = eINSTANCE.getParameterExpression_Parameter();

		/**
		 * The meta object literal for the '{@link poosl.impl.PrimitiveTypeImpl <em>Primitive Type</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.PrimitiveTypeImpl
		 * @see poosl.impl.PooslPackageImpl#getPrimitiveType()
		 * @generated
		 */
		EClass PRIMITIVE_TYPE = eINSTANCE.getPrimitiveType();

		/**
		 * The meta object literal for the '<em><b>Primitive Type Enum</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PRIMITIVE_TYPE__PRIMITIVE_TYPE_ENUM = eINSTANCE.getPrimitiveType_PrimitiveTypeEnum();

		/**
		 * The meta object literal for the '{@link poosl.impl.ProcessClassImpl <em>Process Class</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ProcessClassImpl
		 * @see poosl.impl.PooslPackageImpl#getProcessClass()
		 * @generated
		 */
		EClass PROCESS_CLASS = eINSTANCE.getProcessClass();

		/**
		 * The meta object literal for the '<em><b>Variables</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_CLASS__VARIABLES = eINSTANCE.getProcessClass_Variables();

		/**
		 * The meta object literal for the '<em><b>Methods</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_CLASS__METHODS = eINSTANCE.getProcessClass_Methods();

		/**
		 * The meta object literal for the '<em><b>Super Class</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_CLASS__SUPER_CLASS = eINSTANCE.getProcessClass_SuperClass();

		/**
		 * The meta object literal for the '<em><b>Parameters</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_CLASS__PARAMETERS = eINSTANCE.getProcessClass_Parameters();

		/**
		 * The meta object literal for the '<em><b>Ports</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_CLASS__PORTS = eINSTANCE.getProcessClass_Ports();

		/**
		 * The meta object literal for the '<em><b>Initial Method Call</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_CLASS__INITIAL_METHOD_CALL = eINSTANCE.getProcessClass_InitialMethodCall();

		/**
		 * The meta object literal for the '{@link poosl.impl.ProcessInstanceImpl <em>Process Instance</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ProcessInstanceImpl
		 * @see poosl.impl.PooslPackageImpl#getProcessInstance()
		 * @generated
		 */
		EClass PROCESS_INSTANCE = eINSTANCE.getProcessInstance();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PROCESS_INSTANCE__NAME = eINSTANCE.getProcessInstance_Name();

		/**
		 * The meta object literal for the '<em><b>Class</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_INSTANCE__CLASS = eINSTANCE.getProcessInstance_Class();

		/**
		 * The meta object literal for the '{@link poosl.impl.ProcessMethodImpl <em>Process Method</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ProcessMethodImpl
		 * @see poosl.impl.PooslPackageImpl#getProcessMethod()
		 * @generated
		 */
		EClass PROCESS_METHOD = eINSTANCE.getProcessMethod();

		/**
		 * The meta object literal for the '<em><b>Input Parameters</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_METHOD__INPUT_PARAMETERS = eINSTANCE.getProcessMethod_InputParameters();

		/**
		 * The meta object literal for the '<em><b>Output Parameters</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_METHOD__OUTPUT_PARAMETERS = eINSTANCE.getProcessMethod_OutputParameters();

		/**
		 * The meta object literal for the '<em><b>Variables</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_METHOD__VARIABLES = eINSTANCE.getProcessMethod_Variables();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PROCESS_METHOD__NAME = eINSTANCE.getProcessMethod_Name();

		/**
		 * The meta object literal for the '<em><b>Body</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_METHOD__BODY = eINSTANCE.getProcessMethod_Body();

		/**
		 * The meta object literal for the '{@link poosl.impl.ProcessMethodCallImpl <em>Process Method Call</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ProcessMethodCallImpl
		 * @see poosl.impl.PooslPackageImpl#getProcessMethodCall()
		 * @generated
		 */
		EClass PROCESS_METHOD_CALL = eINSTANCE.getProcessMethodCall();

		/**
		 * The meta object literal for the '<em><b>Method</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_METHOD_CALL__METHOD = eINSTANCE.getProcessMethodCall_Method();

		/**
		 * The meta object literal for the '<em><b>Parameter Expressions</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_METHOD_CALL__PARAMETER_EXPRESSIONS = eINSTANCE.getProcessMethodCall_ParameterExpressions();

		/**
		 * The meta object literal for the '<em><b>Variables</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCESS_METHOD_CALL__VARIABLES = eINSTANCE.getProcessMethodCall_Variables();

		/**
		 * The meta object literal for the '{@link poosl.impl.PortImpl <em>Port</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.PortImpl
		 * @see poosl.impl.PooslPackageImpl#getPort()
		 * @generated
		 */
		EClass PORT = eINSTANCE.getPort();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PORT__NAME = eINSTANCE.getPort_Name();

		/**
		 * The meta object literal for the '{@link poosl.impl.ReferenceSelfImpl <em>Reference Self</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ReferenceSelfImpl
		 * @see poosl.impl.PooslPackageImpl#getReferenceSelf()
		 * @generated
		 */
		EClass REFERENCE_SELF = eINSTANCE.getReferenceSelf();

		/**
		 * The meta object literal for the '{@link poosl.impl.ReturnExpressionImpl <em>Return Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.ReturnExpressionImpl
		 * @see poosl.impl.PooslPackageImpl#getReturnExpression()
		 * @generated
		 */
		EClass RETURN_EXPRESSION = eINSTANCE.getReturnExpression();

		/**
		 * The meta object literal for the '<em><b>Return Expressions</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference RETURN_EXPRESSION__RETURN_EXPRESSIONS = eINSTANCE.getReturnExpression_ReturnExpressions();

		/**
		 * The meta object literal for the '{@link poosl.impl.SelectionImpl <em>Selection</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.SelectionImpl
		 * @see poosl.impl.PooslPackageImpl#getSelection()
		 * @generated
		 */
		EClass SELECTION = eINSTANCE.getSelection();

		/**
		 * The meta object literal for the '<em><b>Statement Lists</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SELECTION__STATEMENT_LISTS = eINSTANCE.getSelection_StatementLists();

		/**
		 * The meta object literal for the '{@link poosl.impl.SkipImpl <em>Skip</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.SkipImpl
		 * @see poosl.impl.PooslPackageImpl#getSkip()
		 * @generated
		 */
		EClass SKIP = eINSTANCE.getSkip();

		/**
		 * The meta object literal for the '{@link poosl.impl.StatementImpl <em>Statement</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.StatementImpl
		 * @see poosl.impl.PooslPackageImpl#getStatement()
		 * @generated
		 */
		EClass STATEMENT = eINSTANCE.getStatement();

		/**
		 * The meta object literal for the '{@link poosl.impl.StatementListImpl <em>Statement List</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.StatementListImpl
		 * @see poosl.impl.PooslPackageImpl#getStatementList()
		 * @generated
		 */
		EClass STATEMENT_LIST = eINSTANCE.getStatementList();

		/**
		 * The meta object literal for the '<em><b>Statements</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference STATEMENT_LIST__STATEMENTS = eINSTANCE.getStatementList_Statements();

		/**
		 * The meta object literal for the '{@link poosl.impl.TimeSynchronisationImpl <em>Time Synchronisation</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.TimeSynchronisationImpl
		 * @see poosl.impl.PooslPackageImpl#getTimeSynchronisation()
		 * @generated
		 */
		EClass TIME_SYNCHRONISATION = eINSTANCE.getTimeSynchronisation();

		/**
		 * The meta object literal for the '<em><b>Duration</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference TIME_SYNCHRONISATION__DURATION = eINSTANCE.getTimeSynchronisation_Duration();

		/**
		 * The meta object literal for the '{@link poosl.impl.TypeImpl <em>Type</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.TypeImpl
		 * @see poosl.impl.PooslPackageImpl#getType()
		 * @generated
		 */
		EClass TYPE = eINSTANCE.getType();

		/**
		 * The meta object literal for the '{@link poosl.impl.VariableImpl <em>Variable</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.VariableImpl
		 * @see poosl.impl.PooslPackageImpl#getVariable()
		 * @generated
		 */
		EClass VARIABLE = eINSTANCE.getVariable();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute VARIABLE__NAME = eINSTANCE.getVariable_Name();

		/**
		 * The meta object literal for the '<em><b>Type</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference VARIABLE__TYPE = eINSTANCE.getVariable_Type();

		/**
		 * The meta object literal for the '{@link poosl.impl.VariableExpressionImpl <em>Variable Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.impl.VariableExpressionImpl
		 * @see poosl.impl.PooslPackageImpl#getVariableExpression()
		 * @generated
		 */
		EClass VARIABLE_EXPRESSION = eINSTANCE.getVariableExpression();

		/**
		 * The meta object literal for the '<em><b>Variable</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference VARIABLE_EXPRESSION__VARIABLE = eINSTANCE.getVariableExpression_Variable();

		/**
		 * The meta object literal for the '{@link poosl.PrimitiveTypeEnum <em>Primitive Type Enum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.PrimitiveTypeEnum
		 * @see poosl.impl.PooslPackageImpl#getPrimitiveTypeEnum()
		 * @generated
		 */
		EEnum PRIMITIVE_TYPE_ENUM = eINSTANCE.getPrimitiveTypeEnum();

		/**
		 * The meta object literal for the '{@link poosl.PrimitiveDataMethodEnum <em>Primitive Data Method Enum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see poosl.PrimitiveDataMethodEnum
		 * @see poosl.impl.PooslPackageImpl#getPrimitiveDataMethodEnum()
		 * @generated
		 */
		EEnum PRIMITIVE_DATA_METHOD_ENUM = eINSTANCE.getPrimitiveDataMethodEnum();

	}

} //PooslPackage
