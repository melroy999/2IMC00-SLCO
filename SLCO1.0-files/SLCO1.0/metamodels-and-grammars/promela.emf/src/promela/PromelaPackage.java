/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;

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
 * @see promela.PromelaFactory
 * @model kind="package"
 * @generated
 */
public interface PromelaPackage extends EPackage {
	/**
	 * The package name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "promela";

	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "promela";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_PREFIX = "promela";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	PromelaPackage eINSTANCE = promela.impl.PromelaPackageImpl.init();

	/**
	 * The meta object id for the '{@link promela.impl.specImpl <em>spec</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.specImpl
	 * @see promela.impl.PromelaPackageImpl#getspec()
	 * @generated
	 */
	int SPEC = 0;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SPEC__NAME = 0;

	/**
	 * The feature id for the '<em><b>Utypes</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SPEC__UTYPES = 1;

	/**
	 * The feature id for the '<em><b>Mtypes</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SPEC__MTYPES = 2;

	/**
	 * The feature id for the '<em><b>One decls</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SPEC__ONE_DECLS = 3;

	/**
	 * The feature id for the '<em><b>Proctypes</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SPEC__PROCTYPES = 4;

	/**
	 * The feature id for the '<em><b>Inits</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SPEC__INITS = 5;

	/**
	 * The feature id for the '<em><b>Nevers</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SPEC__NEVERS = 6;

	/**
	 * The feature id for the '<em><b>Traces</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SPEC__TRACES = 7;

	/**
	 * The feature id for the '<em><b>Primitivetypes</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SPEC__PRIMITIVETYPES = 8;

	/**
	 * The number of structural features of the '<em>spec</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SPEC_FEATURE_COUNT = 9;

	/**
	 * The meta object id for the '{@link promela.impl.proctypeImpl <em>proctype</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.proctypeImpl
	 * @see promela.impl.PromelaPackageImpl#getproctype()
	 * @generated
	 */
	int PROCTYPE = 1;

	/**
	 * The feature id for the '<em><b>Active</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCTYPE__ACTIVE = 0;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCTYPE__NAME = 1;

	/**
	 * The feature id for the '<em><b>One decls</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCTYPE__ONE_DECLS = 2;

	/**
	 * The feature id for the '<em><b>Priority</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCTYPE__PRIORITY = 3;

	/**
	 * The feature id for the '<em><b>Enabler</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCTYPE__ENABLER = 4;

	/**
	 * The feature id for the '<em><b>Sequence</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCTYPE__SEQUENCE = 5;

	/**
	 * The number of structural features of the '<em>proctype</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROCTYPE_FEATURE_COUNT = 6;

	/**
	 * The meta object id for the '{@link promela.impl.initImpl <em>init</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.initImpl
	 * @see promela.impl.PromelaPackageImpl#getinit()
	 * @generated
	 */
	int INIT = 2;

	/**
	 * The feature id for the '<em><b>Priority</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INIT__PRIORITY = 0;

	/**
	 * The feature id for the '<em><b>Sequence</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INIT__SEQUENCE = 1;

	/**
	 * The number of structural features of the '<em>init</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INIT_FEATURE_COUNT = 2;

	/**
	 * The meta object id for the '{@link promela.impl.neverImpl <em>never</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.neverImpl
	 * @see promela.impl.PromelaPackageImpl#getnever()
	 * @generated
	 */
	int NEVER = 3;

	/**
	 * The feature id for the '<em><b>Sequence</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NEVER__SEQUENCE = 0;

	/**
	 * The number of structural features of the '<em>never</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NEVER_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link promela.impl.traceImpl <em>trace</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.traceImpl
	 * @see promela.impl.PromelaPackageImpl#gettrace()
	 * @generated
	 */
	int TRACE = 4;

	/**
	 * The feature id for the '<em><b>Sequence</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRACE__SEQUENCE = 0;

	/**
	 * The number of structural features of the '<em>trace</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRACE_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link promela.impl.utypeImpl <em>utype</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.utypeImpl
	 * @see promela.impl.PromelaPackageImpl#getutype()
	 * @generated
	 */
	int UTYPE = 5;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UTYPE__NAME = 0;

	/**
	 * The feature id for the '<em><b>One decls</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UTYPE__ONE_DECLS = 1;

	/**
	 * The number of structural features of the '<em>utype</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UTYPE_FEATURE_COUNT = 2;

	/**
	 * The meta object id for the '{@link promela.impl.mtypeImpl <em>mtype</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.mtypeImpl
	 * @see promela.impl.PromelaPackageImpl#getmtype()
	 * @generated
	 */
	int MTYPE = 6;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MTYPE__NAME = 0;

	/**
	 * The number of structural features of the '<em>mtype</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MTYPE_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link promela.impl.stepImpl <em>step</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.stepImpl
	 * @see promela.impl.PromelaPackageImpl#getstep()
	 * @generated
	 */
	int STEP = 11;

	/**
	 * The number of structural features of the '<em>step</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STEP_FEATURE_COUNT = 0;

	/**
	 * The meta object id for the '{@link promela.impl.one_declImpl <em>one decl</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.one_declImpl
	 * @see promela.impl.PromelaPackageImpl#getone_decl()
	 * @generated
	 */
	int ONE_DECL = 7;

	/**
	 * The feature id for the '<em><b>Visible</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ONE_DECL__VISIBLE = STEP_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Typename</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ONE_DECL__TYPENAME = STEP_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Ivars</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ONE_DECL__IVARS = STEP_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>one decl</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ONE_DECL_FEATURE_COUNT = STEP_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link promela.impl.typenameImpl <em>typename</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.typenameImpl
	 * @see promela.impl.PromelaPackageImpl#gettypename()
	 * @generated
	 */
	int TYPENAME = 8;

	/**
	 * The number of structural features of the '<em>typename</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TYPENAME_FEATURE_COUNT = 0;

	/**
	 * The meta object id for the '{@link promela.impl.primitivetypeImpl <em>primitivetype</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.primitivetypeImpl
	 * @see promela.impl.PromelaPackageImpl#getprimitivetype()
	 * @generated
	 */
	int PRIMITIVETYPE = 9;

	/**
	 * The feature id for the '<em><b>Primitivetype</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRIMITIVETYPE__PRIMITIVETYPE = TYPENAME_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>primitivetype</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRIMITIVETYPE_FEATURE_COUNT = TYPENAME_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.unameImpl <em>uname</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.unameImpl
	 * @see promela.impl.PromelaPackageImpl#getuname()
	 * @generated
	 */
	int UNAME = 10;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNAME__NAME = TYPENAME_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>uname</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNAME_FEATURE_COUNT = TYPENAME_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.stmntImpl <em>stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getstmnt()
	 * @generated
	 */
	int STMNT = 12;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STMNT__LABEL = STEP_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STMNT__UNLESS = STEP_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STMNT_FEATURE_COUNT = STEP_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link promela.impl.chanassertImpl <em>chanassert</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.chanassertImpl
	 * @see promela.impl.PromelaPackageImpl#getchanassert()
	 * @generated
	 */
	int CHANASSERT = 13;

	/**
	 * The feature id for the '<em><b>Channel assertion</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANASSERT__CHANNEL_ASSERTION = STEP_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Varrefs</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANASSERT__VARREFS = STEP_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>chanassert</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANASSERT_FEATURE_COUNT = STEP_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link promela.impl.ivarImpl <em>ivar</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.ivarImpl
	 * @see promela.impl.PromelaPackageImpl#getivar()
	 * @generated
	 */
	int IVAR = 14;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IVAR__NAME = 0;

	/**
	 * The feature id for the '<em><b>Constant</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IVAR__CONSTANT = 1;

	/**
	 * The feature id for the '<em><b>Any expr or ch init</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IVAR__ANY_EXPR_OR_CH_INIT = 2;

	/**
	 * The number of structural features of the '<em>ivar</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IVAR_FEATURE_COUNT = 3;

	/**
	 * The meta object id for the '{@link promela.impl.any_expr_or_ch_initImpl <em>any expr or ch init</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.any_expr_or_ch_initImpl
	 * @see promela.impl.PromelaPackageImpl#getany_expr_or_ch_init()
	 * @generated
	 */
	int ANY_EXPR_OR_CH_INIT = 15;

	/**
	 * The number of structural features of the '<em>any expr or ch init</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANY_EXPR_OR_CH_INIT_FEATURE_COUNT = 0;

	/**
	 * The meta object id for the '{@link promela.impl.ch_initImpl <em>ch init</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.ch_initImpl
	 * @see promela.impl.PromelaPackageImpl#getch_init()
	 * @generated
	 */
	int CH_INIT = 16;

	/**
	 * The feature id for the '<em><b>Constant</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CH_INIT__CONSTANT = ANY_EXPR_OR_CH_INIT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Typenames</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CH_INIT__TYPENAMES = ANY_EXPR_OR_CH_INIT_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>ch init</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CH_INIT_FEATURE_COUNT = ANY_EXPR_OR_CH_INIT_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link promela.impl.exprImpl <em>expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.exprImpl
	 * @see promela.impl.PromelaPackageImpl#getexpr()
	 * @generated
	 */
	int EXPR = 55;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EXPR__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EXPR__UNLESS = STMNT__UNLESS;

	/**
	 * The number of structural features of the '<em>expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EXPR_FEATURE_COUNT = STMNT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link promela.impl.any_exprImpl <em>any expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.any_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getany_expr()
	 * @generated
	 */
	int ANY_EXPR = 44;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANY_EXPR__LABEL = EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANY_EXPR__UNLESS = EXPR__UNLESS;

	/**
	 * The number of structural features of the '<em>any expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANY_EXPR_FEATURE_COUNT = EXPR_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link promela.impl.varrefImpl <em>varref</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.varrefImpl
	 * @see promela.impl.PromelaPackageImpl#getvarref()
	 * @generated
	 */
	int VARREF = 17;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARREF__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARREF__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Var name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARREF__VAR_NAME = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Any expr</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARREF__ANY_EXPR = ANY_EXPR_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Varref</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARREF__VARREF = ANY_EXPR_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>varref</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARREF_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link promela.impl.sendImpl <em>send</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.sendImpl
	 * @see promela.impl.PromelaPackageImpl#getsend()
	 * @generated
	 */
	int SEND = 18;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SEND__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SEND__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Varref</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SEND__VARREF = STMNT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Send args</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SEND__SEND_ARGS = STMNT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Sendtype</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SEND__SENDTYPE = STMNT_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>send</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SEND_FEATURE_COUNT = STMNT_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link promela.impl.receiveImpl <em>receive</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.receiveImpl
	 * @see promela.impl.PromelaPackageImpl#getreceive()
	 * @generated
	 */
	int RECEIVE = 19;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECEIVE__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECEIVE__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Varref</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECEIVE__VARREF = STMNT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Recv args</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECEIVE__RECV_ARGS = STMNT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Receivetype</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECEIVE__RECEIVETYPE = STMNT_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>receive</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECEIVE_FEATURE_COUNT = STMNT_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link promela.impl.pollImpl <em>poll</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.pollImpl
	 * @see promela.impl.PromelaPackageImpl#getpoll()
	 * @generated
	 */
	int POLL = 20;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int POLL__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int POLL__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Varref</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int POLL__VARREF = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Recv args</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int POLL__RECV_ARGS = ANY_EXPR_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Polltype</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int POLL__POLLTYPE = ANY_EXPR_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>poll</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int POLL_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link promela.impl.recv_argImpl <em>recv arg</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.recv_argImpl
	 * @see promela.impl.PromelaPackageImpl#getrecv_arg()
	 * @generated
	 */
	int RECV_ARG = 21;

	/**
	 * The number of structural features of the '<em>recv arg</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECV_ARG_FEATURE_COUNT = 0;

	/**
	 * The meta object id for the '{@link promela.impl.recv_arg_varrefImpl <em>recv arg varref</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.recv_arg_varrefImpl
	 * @see promela.impl.PromelaPackageImpl#getrecv_arg_varref()
	 * @generated
	 */
	int RECV_ARG_VARREF = 22;

	/**
	 * The feature id for the '<em><b>Varref</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECV_ARG_VARREF__VARREF = RECV_ARG_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>recv arg varref</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECV_ARG_VARREF_FEATURE_COUNT = RECV_ARG_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.recv_arg_eval_exprImpl <em>recv arg eval expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.recv_arg_eval_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getrecv_arg_eval_expr()
	 * @generated
	 */
	int RECV_ARG_EVAL_EXPR = 23;

	/**
	 * The feature id for the '<em><b>Expr</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECV_ARG_EVAL_EXPR__EXPR = RECV_ARG_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>recv arg eval expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECV_ARG_EVAL_EXPR_FEATURE_COUNT = RECV_ARG_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.assignImpl <em>assign</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.assignImpl
	 * @see promela.impl.PromelaPackageImpl#getassign()
	 * @generated
	 */
	int ASSIGN = 24;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN__UNLESS = STMNT__UNLESS;

	/**
	 * The number of structural features of the '<em>assign</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_FEATURE_COUNT = STMNT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link promela.impl.assign_stdImpl <em>assign std</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.assign_stdImpl
	 * @see promela.impl.PromelaPackageImpl#getassign_std()
	 * @generated
	 */
	int ASSIGN_STD = 25;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_STD__LABEL = ASSIGN__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_STD__UNLESS = ASSIGN__UNLESS;

	/**
	 * The feature id for the '<em><b>Varref</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_STD__VARREF = ASSIGN_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Any expr</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_STD__ANY_EXPR = ASSIGN_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>assign std</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_STD_FEATURE_COUNT = ASSIGN_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link promela.impl.assign_incImpl <em>assign inc</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.assign_incImpl
	 * @see promela.impl.PromelaPackageImpl#getassign_inc()
	 * @generated
	 */
	int ASSIGN_INC = 26;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_INC__LABEL = ASSIGN__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_INC__UNLESS = ASSIGN__UNLESS;

	/**
	 * The feature id for the '<em><b>Varref</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_INC__VARREF = ASSIGN_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>assign inc</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_INC_FEATURE_COUNT = ASSIGN_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.assign_decImpl <em>assign dec</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.assign_decImpl
	 * @see promela.impl.PromelaPackageImpl#getassign_dec()
	 * @generated
	 */
	int ASSIGN_DEC = 27;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_DEC__LABEL = ASSIGN__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_DEC__UNLESS = ASSIGN__UNLESS;

	/**
	 * The feature id for the '<em><b>Varref</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_DEC__VARREF = ASSIGN_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>assign dec</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSIGN_DEC_FEATURE_COUNT = ASSIGN_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.if_stmntImpl <em>if stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.if_stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getif_stmnt()
	 * @generated
	 */
	int IF_STMNT = 28;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IF_STMNT__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IF_STMNT__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Options</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IF_STMNT__OPTIONS = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>if stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int IF_STMNT_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.do_stmntImpl <em>do stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.do_stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getdo_stmnt()
	 * @generated
	 */
	int DO_STMNT = 29;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DO_STMNT__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DO_STMNT__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Options</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DO_STMNT__OPTIONS = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>do stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DO_STMNT_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.atomic_stmntImpl <em>atomic stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.atomic_stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getatomic_stmnt()
	 * @generated
	 */
	int ATOMIC_STMNT = 30;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ATOMIC_STMNT__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ATOMIC_STMNT__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Sequence</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ATOMIC_STMNT__SEQUENCE = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>atomic stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ATOMIC_STMNT_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.d_step_stmntImpl <em>dstep stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.d_step_stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getd_step_stmnt()
	 * @generated
	 */
	int DSTEP_STMNT = 31;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DSTEP_STMNT__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DSTEP_STMNT__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Sequence</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DSTEP_STMNT__SEQUENCE = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>dstep stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int DSTEP_STMNT_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.block_stmntImpl <em>block stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.block_stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getblock_stmnt()
	 * @generated
	 */
	int BLOCK_STMNT = 32;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BLOCK_STMNT__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BLOCK_STMNT__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Sequence</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BLOCK_STMNT__SEQUENCE = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>block stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BLOCK_STMNT_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.else_stmntImpl <em>else stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.else_stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getelse_stmnt()
	 * @generated
	 */
	int ELSE_STMNT = 33;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ELSE_STMNT__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ELSE_STMNT__UNLESS = STMNT__UNLESS;

	/**
	 * The number of structural features of the '<em>else stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ELSE_STMNT_FEATURE_COUNT = STMNT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link promela.impl.break_stmntImpl <em>break stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.break_stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getbreak_stmnt()
	 * @generated
	 */
	int BREAK_STMNT = 34;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BREAK_STMNT__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BREAK_STMNT__UNLESS = STMNT__UNLESS;

	/**
	 * The number of structural features of the '<em>break stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BREAK_STMNT_FEATURE_COUNT = STMNT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link promela.impl.goto_stmntImpl <em>goto stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.goto_stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getgoto_stmnt()
	 * @generated
	 */
	int GOTO_STMNT = 35;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int GOTO_STMNT__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int GOTO_STMNT__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Stmnt</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int GOTO_STMNT__STMNT = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>goto stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int GOTO_STMNT_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.print_stmntImpl <em>print stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.print_stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getprint_stmnt()
	 * @generated
	 */
	int PRINT_STMNT = 36;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRINT_STMNT__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRINT_STMNT__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>String</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRINT_STMNT__STRING = STMNT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Arg lst</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRINT_STMNT__ARG_LST = STMNT_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>print stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PRINT_STMNT_FEATURE_COUNT = STMNT_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link promela.impl.assert_stmntImpl <em>assert stmnt</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.assert_stmntImpl
	 * @see promela.impl.PromelaPackageImpl#getassert_stmnt()
	 * @generated
	 */
	int ASSERT_STMNT = 37;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSERT_STMNT__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSERT_STMNT__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Expr</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSERT_STMNT__EXPR = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>assert stmnt</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ASSERT_STMNT_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.c_codeImpl <em>ccode</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.c_codeImpl
	 * @see promela.impl.PromelaPackageImpl#getc_code()
	 * @generated
	 */
	int CCODE = 38;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CCODE__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CCODE__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Embedded ccode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CCODE__EMBEDDED_CCODE = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>ccode</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CCODE_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.c_exprImpl <em>cexpr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.c_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getc_expr()
	 * @generated
	 */
	int CEXPR = 39;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CEXPR__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CEXPR__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Embedded ccode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CEXPR__EMBEDDED_CCODE = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>cexpr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CEXPR_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.c_declImpl <em>cdecl</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.c_declImpl
	 * @see promela.impl.PromelaPackageImpl#getc_decl()
	 * @generated
	 */
	int CDECL = 40;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CDECL__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CDECL__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Embedded ccode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CDECL__EMBEDDED_CCODE = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>cdecl</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CDECL_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.c_trackImpl <em>ctrack</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.c_trackImpl
	 * @see promela.impl.PromelaPackageImpl#getc_track()
	 * @generated
	 */
	int CTRACK = 41;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CTRACK__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CTRACK__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Embedded ccode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CTRACK__EMBEDDED_CCODE = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>ctrack</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CTRACK_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.c_stateImpl <em>cstate</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.c_stateImpl
	 * @see promela.impl.PromelaPackageImpl#getc_state()
	 * @generated
	 */
	int CSTATE = 42;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CSTATE__LABEL = STMNT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CSTATE__UNLESS = STMNT__UNLESS;

	/**
	 * The feature id for the '<em><b>Embedded ccode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CSTATE__EMBEDDED_CCODE = STMNT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>cstate</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CSTATE_FEATURE_COUNT = STMNT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.optionsImpl <em>options</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.optionsImpl
	 * @see promela.impl.PromelaPackageImpl#getoptions()
	 * @generated
	 */
	int OPTIONS = 43;

	/**
	 * The feature id for the '<em><b>Sequence</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OPTIONS__SEQUENCE = 0;

	/**
	 * The number of structural features of the '<em>options</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OPTIONS_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link promela.impl.bin_exprImpl <em>bin expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.bin_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getbin_expr()
	 * @generated
	 */
	int BIN_EXPR = 45;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIN_EXPR__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIN_EXPR__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Operator</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIN_EXPR__OPERATOR = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Operand1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIN_EXPR__OPERAND1 = ANY_EXPR_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Operand2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIN_EXPR__OPERAND2 = ANY_EXPR_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>bin expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIN_EXPR_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link promela.impl.un_exprImpl <em>un expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.un_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getun_expr()
	 * @generated
	 */
	int UN_EXPR = 46;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UN_EXPR__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UN_EXPR__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Operator</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UN_EXPR__OPERATOR = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Operand</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UN_EXPR__OPERAND = ANY_EXPR_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>un expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UN_EXPR_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link promela.impl.cond_exprImpl <em>cond expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.cond_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getcond_expr()
	 * @generated
	 */
	int COND_EXPR = 47;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COND_EXPR__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COND_EXPR__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Expr1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COND_EXPR__EXPR1 = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Expr2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COND_EXPR__EXPR2 = ANY_EXPR_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Expr3</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COND_EXPR__EXPR3 = ANY_EXPR_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>cond expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COND_EXPR_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link promela.impl.len_exprImpl <em>len expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.len_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getlen_expr()
	 * @generated
	 */
	int LEN_EXPR = 48;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LEN_EXPR__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LEN_EXPR__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Varref</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LEN_EXPR__VARREF = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>len expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LEN_EXPR_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.timeout_exprImpl <em>timeout expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.timeout_exprImpl
	 * @see promela.impl.PromelaPackageImpl#gettimeout_expr()
	 * @generated
	 */
	int TIMEOUT_EXPR = 49;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TIMEOUT_EXPR__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TIMEOUT_EXPR__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The number of structural features of the '<em>timeout expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TIMEOUT_EXPR_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link promela.impl.np__exprImpl <em>np expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.np__exprImpl
	 * @see promela.impl.PromelaPackageImpl#getnp__expr()
	 * @generated
	 */
	int NP_EXPR = 50;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NP_EXPR__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NP_EXPR__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The number of structural features of the '<em>np expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NP_EXPR_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link promela.impl.enabled_exprImpl <em>enabled expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.enabled_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getenabled_expr()
	 * @generated
	 */
	int ENABLED_EXPR = 51;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ENABLED_EXPR__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ENABLED_EXPR__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Any expr</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ENABLED_EXPR__ANY_EXPR = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>enabled expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ENABLED_EXPR_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.pc_value_exprImpl <em>pc value expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.pc_value_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getpc_value_expr()
	 * @generated
	 */
	int PC_VALUE_EXPR = 52;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PC_VALUE_EXPR__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PC_VALUE_EXPR__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Any expr</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PC_VALUE_EXPR__ANY_EXPR = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>pc value expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PC_VALUE_EXPR_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.name_exprImpl <em>name expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.name_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getname_expr()
	 * @generated
	 */
	int NAME_EXPR = 53;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NAME_EXPR__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NAME_EXPR__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NAME_EXPR__NAME = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Any expr</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NAME_EXPR__ANY_EXPR = ANY_EXPR_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>At name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NAME_EXPR__AT_NAME = ANY_EXPR_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>name expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NAME_EXPR_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link promela.impl.run_exprImpl <em>run expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.run_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getrun_expr()
	 * @generated
	 */
	int RUN_EXPR = 54;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RUN_EXPR__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RUN_EXPR__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Proc</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RUN_EXPR__PROC = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Arg lst</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RUN_EXPR__ARG_LST = ANY_EXPR_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Priority</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RUN_EXPR__PRIORITY = ANY_EXPR_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>run expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RUN_EXPR_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link promela.impl.andor_exprImpl <em>andor expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.andor_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getandor_expr()
	 * @generated
	 */
	int ANDOR_EXPR = 56;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANDOR_EXPR__LABEL = EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANDOR_EXPR__UNLESS = EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Operator</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANDOR_EXPR__OPERATOR = EXPR_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Operand1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANDOR_EXPR__OPERAND1 = EXPR_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Operand2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANDOR_EXPR__OPERAND2 = EXPR_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>andor expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANDOR_EXPR_FEATURE_COUNT = EXPR_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link promela.impl.chanpoll_exprImpl <em>chanpoll expr</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.chanpoll_exprImpl
	 * @see promela.impl.PromelaPackageImpl#getchanpoll_expr()
	 * @generated
	 */
	int CHANPOLL_EXPR = 57;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANPOLL_EXPR__LABEL = EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANPOLL_EXPR__UNLESS = EXPR__UNLESS;

	/**
	 * The feature id for the '<em><b>Chan poll</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANPOLL_EXPR__CHAN_POLL = EXPR_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Varref</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANPOLL_EXPR__VARREF = EXPR_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>chanpoll expr</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANPOLL_EXPR_FEATURE_COUNT = EXPR_FEATURE_COUNT + 2;

	/**
	 * The meta object id for the '{@link promela.impl.constantImpl <em>constant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.constantImpl
	 * @see promela.impl.PromelaPackageImpl#getconstant()
	 * @generated
	 */
	int CONSTANT = 58;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT__LABEL = ANY_EXPR__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT__UNLESS = ANY_EXPR__UNLESS;

	/**
	 * The number of structural features of the '<em>constant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT_FEATURE_COUNT = ANY_EXPR_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link promela.impl.num_constImpl <em>num const</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.num_constImpl
	 * @see promela.impl.PromelaPackageImpl#getnum_const()
	 * @generated
	 */
	int NUM_CONST = 59;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NUM_CONST__LABEL = CONSTANT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NUM_CONST__UNLESS = CONSTANT__UNLESS;

	/**
	 * The feature id for the '<em><b>Number</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NUM_CONST__NUMBER = CONSTANT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>num const</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int NUM_CONST_FEATURE_COUNT = CONSTANT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.enum_constImpl <em>enum const</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.enum_constImpl
	 * @see promela.impl.PromelaPackageImpl#getenum_const()
	 * @generated
	 */
	int ENUM_CONST = 60;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ENUM_CONST__LABEL = CONSTANT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ENUM_CONST__UNLESS = CONSTANT__UNLESS;

	/**
	 * The feature id for the '<em><b>Const enum</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ENUM_CONST__CONST_ENUM = CONSTANT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>enum const</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ENUM_CONST_FEATURE_COUNT = CONSTANT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.impl.mtype_constImpl <em>mtype const</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.impl.mtype_constImpl
	 * @see promela.impl.PromelaPackageImpl#getmtype_const()
	 * @generated
	 */
	int MTYPE_CONST = 61;

	/**
	 * The feature id for the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MTYPE_CONST__LABEL = CONSTANT__LABEL;

	/**
	 * The feature id for the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MTYPE_CONST__UNLESS = CONSTANT__UNLESS;

	/**
	 * The feature id for the '<em><b>Mtype</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MTYPE_CONST__MTYPE = CONSTANT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>mtype const</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MTYPE_CONST_FEATURE_COUNT = CONSTANT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link promela.visibleenum <em>visibleenum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.visibleenum
	 * @see promela.impl.PromelaPackageImpl#getvisibleenum()
	 * @generated
	 */
	int VISIBLEENUM = 62;

	/**
	 * The meta object id for the '{@link promela.primitivetypeenum <em>primitivetypeenum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.primitivetypeenum
	 * @see promela.impl.PromelaPackageImpl#getprimitivetypeenum()
	 * @generated
	 */
	int PRIMITIVETYPEENUM = 63;

	/**
	 * The meta object id for the '{@link promela.channelassertion <em>channelassertion</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.channelassertion
	 * @see promela.impl.PromelaPackageImpl#getchannelassertion()
	 * @generated
	 */
	int CHANNELASSERTION = 64;

	/**
	 * The meta object id for the '{@link promela.sendtypeenum <em>sendtypeenum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.sendtypeenum
	 * @see promela.impl.PromelaPackageImpl#getsendtypeenum()
	 * @generated
	 */
	int SENDTYPEENUM = 65;

	/**
	 * The meta object id for the '{@link promela.receivetypeenum <em>receivetypeenum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.receivetypeenum
	 * @see promela.impl.PromelaPackageImpl#getreceivetypeenum()
	 * @generated
	 */
	int RECEIVETYPEENUM = 66;

	/**
	 * The meta object id for the '{@link promela.polltypeenum <em>polltypeenum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.polltypeenum
	 * @see promela.impl.PromelaPackageImpl#getpolltypeenum()
	 * @generated
	 */
	int POLLTYPEENUM = 67;

	/**
	 * The meta object id for the '{@link promela.andor <em>andor</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.andor
	 * @see promela.impl.PromelaPackageImpl#getandor()
	 * @generated
	 */
	int ANDOR = 68;

	/**
	 * The meta object id for the '{@link promela.binarop <em>binarop</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.binarop
	 * @see promela.impl.PromelaPackageImpl#getbinarop()
	 * @generated
	 */
	int BINAROP = 69;

	/**
	 * The meta object id for the '{@link promela.unarop <em>unarop</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.unarop
	 * @see promela.impl.PromelaPackageImpl#getunarop()
	 * @generated
	 */
	int UNAROP = 70;

	/**
	 * The meta object id for the '{@link promela.chanpoll <em>chanpoll</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.chanpoll
	 * @see promela.impl.PromelaPackageImpl#getchanpoll()
	 * @generated
	 */
	int CHANPOLL = 71;

	/**
	 * The meta object id for the '{@link promela.constenum <em>constenum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see promela.constenum
	 * @see promela.impl.PromelaPackageImpl#getconstenum()
	 * @generated
	 */
	int CONSTENUM = 72;


	/**
	 * Returns the meta object for class '{@link promela.spec <em>spec</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>spec</em>'.
	 * @see promela.spec
	 * @generated
	 */
	EClass getspec();

	/**
	 * Returns the meta object for the attribute '{@link promela.spec#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see promela.spec#getName()
	 * @see #getspec()
	 * @generated
	 */
	EAttribute getspec_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.spec#getUtypes <em>Utypes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Utypes</em>'.
	 * @see promela.spec#getUtypes()
	 * @see #getspec()
	 * @generated
	 */
	EReference getspec_Utypes();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.spec#getMtypes <em>Mtypes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Mtypes</em>'.
	 * @see promela.spec#getMtypes()
	 * @see #getspec()
	 * @generated
	 */
	EReference getspec_Mtypes();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.spec#getOne_decls <em>One decls</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>One decls</em>'.
	 * @see promela.spec#getOne_decls()
	 * @see #getspec()
	 * @generated
	 */
	EReference getspec_One_decls();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.spec#getProctypes <em>Proctypes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Proctypes</em>'.
	 * @see promela.spec#getProctypes()
	 * @see #getspec()
	 * @generated
	 */
	EReference getspec_Proctypes();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.spec#getInits <em>Inits</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Inits</em>'.
	 * @see promela.spec#getInits()
	 * @see #getspec()
	 * @generated
	 */
	EReference getspec_Inits();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.spec#getNevers <em>Nevers</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Nevers</em>'.
	 * @see promela.spec#getNevers()
	 * @see #getspec()
	 * @generated
	 */
	EReference getspec_Nevers();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.spec#getTraces <em>Traces</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Traces</em>'.
	 * @see promela.spec#getTraces()
	 * @see #getspec()
	 * @generated
	 */
	EReference getspec_Traces();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.spec#getPrimitivetypes <em>Primitivetypes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Primitivetypes</em>'.
	 * @see promela.spec#getPrimitivetypes()
	 * @see #getspec()
	 * @generated
	 */
	EReference getspec_Primitivetypes();

	/**
	 * Returns the meta object for class '{@link promela.proctype <em>proctype</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>proctype</em>'.
	 * @see promela.proctype
	 * @generated
	 */
	EClass getproctype();

	/**
	 * Returns the meta object for the attribute '{@link promela.proctype#getActive <em>Active</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Active</em>'.
	 * @see promela.proctype#getActive()
	 * @see #getproctype()
	 * @generated
	 */
	EAttribute getproctype_Active();

	/**
	 * Returns the meta object for the attribute '{@link promela.proctype#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see promela.proctype#getName()
	 * @see #getproctype()
	 * @generated
	 */
	EAttribute getproctype_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.proctype#getOne_decls <em>One decls</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>One decls</em>'.
	 * @see promela.proctype#getOne_decls()
	 * @see #getproctype()
	 * @generated
	 */
	EReference getproctype_One_decls();

	/**
	 * Returns the meta object for the attribute '{@link promela.proctype#getPriority <em>Priority</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Priority</em>'.
	 * @see promela.proctype#getPriority()
	 * @see #getproctype()
	 * @generated
	 */
	EAttribute getproctype_Priority();

	/**
	 * Returns the meta object for the containment reference '{@link promela.proctype#getEnabler <em>Enabler</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Enabler</em>'.
	 * @see promela.proctype#getEnabler()
	 * @see #getproctype()
	 * @generated
	 */
	EReference getproctype_Enabler();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.proctype#getSequence <em>Sequence</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Sequence</em>'.
	 * @see promela.proctype#getSequence()
	 * @see #getproctype()
	 * @generated
	 */
	EReference getproctype_Sequence();

	/**
	 * Returns the meta object for class '{@link promela.init <em>init</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>init</em>'.
	 * @see promela.init
	 * @generated
	 */
	EClass getinit();

	/**
	 * Returns the meta object for the attribute '{@link promela.init#getPriority <em>Priority</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Priority</em>'.
	 * @see promela.init#getPriority()
	 * @see #getinit()
	 * @generated
	 */
	EAttribute getinit_Priority();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.init#getSequence <em>Sequence</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Sequence</em>'.
	 * @see promela.init#getSequence()
	 * @see #getinit()
	 * @generated
	 */
	EReference getinit_Sequence();

	/**
	 * Returns the meta object for class '{@link promela.never <em>never</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>never</em>'.
	 * @see promela.never
	 * @generated
	 */
	EClass getnever();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.never#getSequence <em>Sequence</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Sequence</em>'.
	 * @see promela.never#getSequence()
	 * @see #getnever()
	 * @generated
	 */
	EReference getnever_Sequence();

	/**
	 * Returns the meta object for class '{@link promela.trace <em>trace</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>trace</em>'.
	 * @see promela.trace
	 * @generated
	 */
	EClass gettrace();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.trace#getSequence <em>Sequence</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Sequence</em>'.
	 * @see promela.trace#getSequence()
	 * @see #gettrace()
	 * @generated
	 */
	EReference gettrace_Sequence();

	/**
	 * Returns the meta object for class '{@link promela.utype <em>utype</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>utype</em>'.
	 * @see promela.utype
	 * @generated
	 */
	EClass getutype();

	/**
	 * Returns the meta object for the attribute '{@link promela.utype#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see promela.utype#getName()
	 * @see #getutype()
	 * @generated
	 */
	EAttribute getutype_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.utype#getOne_decls <em>One decls</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>One decls</em>'.
	 * @see promela.utype#getOne_decls()
	 * @see #getutype()
	 * @generated
	 */
	EReference getutype_One_decls();

	/**
	 * Returns the meta object for class '{@link promela.mtype <em>mtype</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>mtype</em>'.
	 * @see promela.mtype
	 * @generated
	 */
	EClass getmtype();

	/**
	 * Returns the meta object for the attribute '{@link promela.mtype#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see promela.mtype#getName()
	 * @see #getmtype()
	 * @generated
	 */
	EAttribute getmtype_Name();

	/**
	 * Returns the meta object for class '{@link promela.one_decl <em>one decl</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>one decl</em>'.
	 * @see promela.one_decl
	 * @generated
	 */
	EClass getone_decl();

	/**
	 * Returns the meta object for the attribute '{@link promela.one_decl#getVisible <em>Visible</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Visible</em>'.
	 * @see promela.one_decl#getVisible()
	 * @see #getone_decl()
	 * @generated
	 */
	EAttribute getone_decl_Visible();

	/**
	 * Returns the meta object for the reference '{@link promela.one_decl#getTypename <em>Typename</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Typename</em>'.
	 * @see promela.one_decl#getTypename()
	 * @see #getone_decl()
	 * @generated
	 */
	EReference getone_decl_Typename();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.one_decl#getIvars <em>Ivars</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Ivars</em>'.
	 * @see promela.one_decl#getIvars()
	 * @see #getone_decl()
	 * @generated
	 */
	EReference getone_decl_Ivars();

	/**
	 * Returns the meta object for class '{@link promela.typename <em>typename</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>typename</em>'.
	 * @see promela.typename
	 * @generated
	 */
	EClass gettypename();

	/**
	 * Returns the meta object for class '{@link promela.primitivetype <em>primitivetype</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>primitivetype</em>'.
	 * @see promela.primitivetype
	 * @generated
	 */
	EClass getprimitivetype();

	/**
	 * Returns the meta object for the attribute '{@link promela.primitivetype#getPrimitivetype <em>Primitivetype</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Primitivetype</em>'.
	 * @see promela.primitivetype#getPrimitivetype()
	 * @see #getprimitivetype()
	 * @generated
	 */
	EAttribute getprimitivetype_Primitivetype();

	/**
	 * Returns the meta object for class '{@link promela.uname <em>uname</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>uname</em>'.
	 * @see promela.uname
	 * @generated
	 */
	EClass getuname();

	/**
	 * Returns the meta object for the attribute '{@link promela.uname#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see promela.uname#getName()
	 * @see #getuname()
	 * @generated
	 */
	EAttribute getuname_Name();

	/**
	 * Returns the meta object for class '{@link promela.step <em>step</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>step</em>'.
	 * @see promela.step
	 * @generated
	 */
	EClass getstep();

	/**
	 * Returns the meta object for class '{@link promela.stmnt <em>stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>stmnt</em>'.
	 * @see promela.stmnt
	 * @generated
	 */
	EClass getstmnt();

	/**
	 * Returns the meta object for the attribute '{@link promela.stmnt#getLabel <em>Label</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Label</em>'.
	 * @see promela.stmnt#getLabel()
	 * @see #getstmnt()
	 * @generated
	 */
	EAttribute getstmnt_Label();

	/**
	 * Returns the meta object for the containment reference '{@link promela.stmnt#getUnless <em>Unless</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Unless</em>'.
	 * @see promela.stmnt#getUnless()
	 * @see #getstmnt()
	 * @generated
	 */
	EReference getstmnt_Unless();

	/**
	 * Returns the meta object for class '{@link promela.chanassert <em>chanassert</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>chanassert</em>'.
	 * @see promela.chanassert
	 * @generated
	 */
	EClass getchanassert();

	/**
	 * Returns the meta object for the attribute '{@link promela.chanassert#getChannel_assertion <em>Channel assertion</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Channel assertion</em>'.
	 * @see promela.chanassert#getChannel_assertion()
	 * @see #getchanassert()
	 * @generated
	 */
	EAttribute getchanassert_Channel_assertion();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.chanassert#getVarrefs <em>Varrefs</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Varrefs</em>'.
	 * @see promela.chanassert#getVarrefs()
	 * @see #getchanassert()
	 * @generated
	 */
	EReference getchanassert_Varrefs();

	/**
	 * Returns the meta object for class '{@link promela.ivar <em>ivar</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>ivar</em>'.
	 * @see promela.ivar
	 * @generated
	 */
	EClass getivar();

	/**
	 * Returns the meta object for the attribute '{@link promela.ivar#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see promela.ivar#getName()
	 * @see #getivar()
	 * @generated
	 */
	EAttribute getivar_Name();

	/**
	 * Returns the meta object for the attribute '{@link promela.ivar#getConstant <em>Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Constant</em>'.
	 * @see promela.ivar#getConstant()
	 * @see #getivar()
	 * @generated
	 */
	EAttribute getivar_Constant();

	/**
	 * Returns the meta object for the containment reference '{@link promela.ivar#getAny_expr_or_ch_init <em>Any expr or ch init</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Any expr or ch init</em>'.
	 * @see promela.ivar#getAny_expr_or_ch_init()
	 * @see #getivar()
	 * @generated
	 */
	EReference getivar_Any_expr_or_ch_init();

	/**
	 * Returns the meta object for class '{@link promela.any_expr_or_ch_init <em>any expr or ch init</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>any expr or ch init</em>'.
	 * @see promela.any_expr_or_ch_init
	 * @generated
	 */
	EClass getany_expr_or_ch_init();

	/**
	 * Returns the meta object for class '{@link promela.ch_init <em>ch init</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>ch init</em>'.
	 * @see promela.ch_init
	 * @generated
	 */
	EClass getch_init();

	/**
	 * Returns the meta object for the attribute '{@link promela.ch_init#getConstant <em>Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Constant</em>'.
	 * @see promela.ch_init#getConstant()
	 * @see #getch_init()
	 * @generated
	 */
	EAttribute getch_init_Constant();

	/**
	 * Returns the meta object for the reference list '{@link promela.ch_init#getTypenames <em>Typenames</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Typenames</em>'.
	 * @see promela.ch_init#getTypenames()
	 * @see #getch_init()
	 * @generated
	 */
	EReference getch_init_Typenames();

	/**
	 * Returns the meta object for class '{@link promela.varref <em>varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>varref</em>'.
	 * @see promela.varref
	 * @generated
	 */
	EClass getvarref();

	/**
	 * Returns the meta object for the attribute '{@link promela.varref#getVar_name <em>Var name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Var name</em>'.
	 * @see promela.varref#getVar_name()
	 * @see #getvarref()
	 * @generated
	 */
	EAttribute getvarref_Var_name();

	/**
	 * Returns the meta object for the reference '{@link promela.varref#getAny_expr <em>Any expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Any expr</em>'.
	 * @see promela.varref#getAny_expr()
	 * @see #getvarref()
	 * @generated
	 */
	EReference getvarref_Any_expr();

	/**
	 * Returns the meta object for the reference '{@link promela.varref#getVarref <em>Varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Varref</em>'.
	 * @see promela.varref#getVarref()
	 * @see #getvarref()
	 * @generated
	 */
	EReference getvarref_Varref();

	/**
	 * Returns the meta object for class '{@link promela.send <em>send</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>send</em>'.
	 * @see promela.send
	 * @generated
	 */
	EClass getsend();

	/**
	 * Returns the meta object for the containment reference '{@link promela.send#getVarref <em>Varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Varref</em>'.
	 * @see promela.send#getVarref()
	 * @see #getsend()
	 * @generated
	 */
	EReference getsend_Varref();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.send#getSend_args <em>Send args</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Send args</em>'.
	 * @see promela.send#getSend_args()
	 * @see #getsend()
	 * @generated
	 */
	EReference getsend_Send_args();

	/**
	 * Returns the meta object for the attribute '{@link promela.send#getSendtype <em>Sendtype</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Sendtype</em>'.
	 * @see promela.send#getSendtype()
	 * @see #getsend()
	 * @generated
	 */
	EAttribute getsend_Sendtype();

	/**
	 * Returns the meta object for class '{@link promela.receive <em>receive</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>receive</em>'.
	 * @see promela.receive
	 * @generated
	 */
	EClass getreceive();

	/**
	 * Returns the meta object for the containment reference '{@link promela.receive#getVarref <em>Varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Varref</em>'.
	 * @see promela.receive#getVarref()
	 * @see #getreceive()
	 * @generated
	 */
	EReference getreceive_Varref();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.receive#getRecv_args <em>Recv args</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Recv args</em>'.
	 * @see promela.receive#getRecv_args()
	 * @see #getreceive()
	 * @generated
	 */
	EReference getreceive_Recv_args();

	/**
	 * Returns the meta object for the attribute '{@link promela.receive#getReceivetype <em>Receivetype</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Receivetype</em>'.
	 * @see promela.receive#getReceivetype()
	 * @see #getreceive()
	 * @generated
	 */
	EAttribute getreceive_Receivetype();

	/**
	 * Returns the meta object for class '{@link promela.poll <em>poll</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>poll</em>'.
	 * @see promela.poll
	 * @generated
	 */
	EClass getpoll();

	/**
	 * Returns the meta object for the containment reference '{@link promela.poll#getVarref <em>Varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Varref</em>'.
	 * @see promela.poll#getVarref()
	 * @see #getpoll()
	 * @generated
	 */
	EReference getpoll_Varref();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.poll#getRecv_args <em>Recv args</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Recv args</em>'.
	 * @see promela.poll#getRecv_args()
	 * @see #getpoll()
	 * @generated
	 */
	EReference getpoll_Recv_args();

	/**
	 * Returns the meta object for the attribute '{@link promela.poll#getPolltype <em>Polltype</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Polltype</em>'.
	 * @see promela.poll#getPolltype()
	 * @see #getpoll()
	 * @generated
	 */
	EAttribute getpoll_Polltype();

	/**
	 * Returns the meta object for class '{@link promela.recv_arg <em>recv arg</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>recv arg</em>'.
	 * @see promela.recv_arg
	 * @generated
	 */
	EClass getrecv_arg();

	/**
	 * Returns the meta object for class '{@link promela.recv_arg_varref <em>recv arg varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>recv arg varref</em>'.
	 * @see promela.recv_arg_varref
	 * @generated
	 */
	EClass getrecv_arg_varref();

	/**
	 * Returns the meta object for the containment reference '{@link promela.recv_arg_varref#getVarref <em>Varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Varref</em>'.
	 * @see promela.recv_arg_varref#getVarref()
	 * @see #getrecv_arg_varref()
	 * @generated
	 */
	EReference getrecv_arg_varref_Varref();

	/**
	 * Returns the meta object for class '{@link promela.recv_arg_eval_expr <em>recv arg eval expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>recv arg eval expr</em>'.
	 * @see promela.recv_arg_eval_expr
	 * @generated
	 */
	EClass getrecv_arg_eval_expr();

	/**
	 * Returns the meta object for the containment reference '{@link promela.recv_arg_eval_expr#getExpr <em>Expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Expr</em>'.
	 * @see promela.recv_arg_eval_expr#getExpr()
	 * @see #getrecv_arg_eval_expr()
	 * @generated
	 */
	EReference getrecv_arg_eval_expr_Expr();

	/**
	 * Returns the meta object for class '{@link promela.assign <em>assign</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>assign</em>'.
	 * @see promela.assign
	 * @generated
	 */
	EClass getassign();

	/**
	 * Returns the meta object for class '{@link promela.assign_std <em>assign std</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>assign std</em>'.
	 * @see promela.assign_std
	 * @generated
	 */
	EClass getassign_std();

	/**
	 * Returns the meta object for the containment reference '{@link promela.assign_std#getVarref <em>Varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Varref</em>'.
	 * @see promela.assign_std#getVarref()
	 * @see #getassign_std()
	 * @generated
	 */
	EReference getassign_std_Varref();

	/**
	 * Returns the meta object for the containment reference '{@link promela.assign_std#getAny_expr <em>Any expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Any expr</em>'.
	 * @see promela.assign_std#getAny_expr()
	 * @see #getassign_std()
	 * @generated
	 */
	EReference getassign_std_Any_expr();

	/**
	 * Returns the meta object for class '{@link promela.assign_inc <em>assign inc</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>assign inc</em>'.
	 * @see promela.assign_inc
	 * @generated
	 */
	EClass getassign_inc();

	/**
	 * Returns the meta object for the containment reference '{@link promela.assign_inc#getVarref <em>Varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Varref</em>'.
	 * @see promela.assign_inc#getVarref()
	 * @see #getassign_inc()
	 * @generated
	 */
	EReference getassign_inc_Varref();

	/**
	 * Returns the meta object for class '{@link promela.assign_dec <em>assign dec</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>assign dec</em>'.
	 * @see promela.assign_dec
	 * @generated
	 */
	EClass getassign_dec();

	/**
	 * Returns the meta object for the containment reference '{@link promela.assign_dec#getVarref <em>Varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Varref</em>'.
	 * @see promela.assign_dec#getVarref()
	 * @see #getassign_dec()
	 * @generated
	 */
	EReference getassign_dec_Varref();

	/**
	 * Returns the meta object for class '{@link promela.if_stmnt <em>if stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>if stmnt</em>'.
	 * @see promela.if_stmnt
	 * @generated
	 */
	EClass getif_stmnt();

	/**
	 * Returns the meta object for the containment reference '{@link promela.if_stmnt#getOptions <em>Options</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Options</em>'.
	 * @see promela.if_stmnt#getOptions()
	 * @see #getif_stmnt()
	 * @generated
	 */
	EReference getif_stmnt_Options();

	/**
	 * Returns the meta object for class '{@link promela.do_stmnt <em>do stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>do stmnt</em>'.
	 * @see promela.do_stmnt
	 * @generated
	 */
	EClass getdo_stmnt();

	/**
	 * Returns the meta object for the containment reference '{@link promela.do_stmnt#getOptions <em>Options</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Options</em>'.
	 * @see promela.do_stmnt#getOptions()
	 * @see #getdo_stmnt()
	 * @generated
	 */
	EReference getdo_stmnt_Options();

	/**
	 * Returns the meta object for class '{@link promela.atomic_stmnt <em>atomic stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>atomic stmnt</em>'.
	 * @see promela.atomic_stmnt
	 * @generated
	 */
	EClass getatomic_stmnt();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.atomic_stmnt#getSequence <em>Sequence</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Sequence</em>'.
	 * @see promela.atomic_stmnt#getSequence()
	 * @see #getatomic_stmnt()
	 * @generated
	 */
	EReference getatomic_stmnt_Sequence();

	/**
	 * Returns the meta object for class '{@link promela.d_step_stmnt <em>dstep stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>dstep stmnt</em>'.
	 * @see promela.d_step_stmnt
	 * @generated
	 */
	EClass getd_step_stmnt();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.d_step_stmnt#getSequence <em>Sequence</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Sequence</em>'.
	 * @see promela.d_step_stmnt#getSequence()
	 * @see #getd_step_stmnt()
	 * @generated
	 */
	EReference getd_step_stmnt_Sequence();

	/**
	 * Returns the meta object for class '{@link promela.block_stmnt <em>block stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>block stmnt</em>'.
	 * @see promela.block_stmnt
	 * @generated
	 */
	EClass getblock_stmnt();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.block_stmnt#getSequence <em>Sequence</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Sequence</em>'.
	 * @see promela.block_stmnt#getSequence()
	 * @see #getblock_stmnt()
	 * @generated
	 */
	EReference getblock_stmnt_Sequence();

	/**
	 * Returns the meta object for class '{@link promela.else_stmnt <em>else stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>else stmnt</em>'.
	 * @see promela.else_stmnt
	 * @generated
	 */
	EClass getelse_stmnt();

	/**
	 * Returns the meta object for class '{@link promela.break_stmnt <em>break stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>break stmnt</em>'.
	 * @see promela.break_stmnt
	 * @generated
	 */
	EClass getbreak_stmnt();

	/**
	 * Returns the meta object for class '{@link promela.goto_stmnt <em>goto stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>goto stmnt</em>'.
	 * @see promela.goto_stmnt
	 * @generated
	 */
	EClass getgoto_stmnt();

	/**
	 * Returns the meta object for the reference '{@link promela.goto_stmnt#getStmnt <em>Stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Stmnt</em>'.
	 * @see promela.goto_stmnt#getStmnt()
	 * @see #getgoto_stmnt()
	 * @generated
	 */
	EReference getgoto_stmnt_Stmnt();

	/**
	 * Returns the meta object for class '{@link promela.print_stmnt <em>print stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>print stmnt</em>'.
	 * @see promela.print_stmnt
	 * @generated
	 */
	EClass getprint_stmnt();

	/**
	 * Returns the meta object for the attribute '{@link promela.print_stmnt#getString <em>String</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>String</em>'.
	 * @see promela.print_stmnt#getString()
	 * @see #getprint_stmnt()
	 * @generated
	 */
	EAttribute getprint_stmnt_String();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.print_stmnt#getArg_lst <em>Arg lst</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Arg lst</em>'.
	 * @see promela.print_stmnt#getArg_lst()
	 * @see #getprint_stmnt()
	 * @generated
	 */
	EReference getprint_stmnt_Arg_lst();

	/**
	 * Returns the meta object for class '{@link promela.assert_stmnt <em>assert stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>assert stmnt</em>'.
	 * @see promela.assert_stmnt
	 * @generated
	 */
	EClass getassert_stmnt();

	/**
	 * Returns the meta object for the containment reference '{@link promela.assert_stmnt#getExpr <em>Expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Expr</em>'.
	 * @see promela.assert_stmnt#getExpr()
	 * @see #getassert_stmnt()
	 * @generated
	 */
	EReference getassert_stmnt_Expr();

	/**
	 * Returns the meta object for class '{@link promela.c_code <em>ccode</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>ccode</em>'.
	 * @see promela.c_code
	 * @generated
	 */
	EClass getc_code();

	/**
	 * Returns the meta object for the attribute '{@link promela.c_code#getEmbedded_c_code <em>Embedded ccode</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Embedded ccode</em>'.
	 * @see promela.c_code#getEmbedded_c_code()
	 * @see #getc_code()
	 * @generated
	 */
	EAttribute getc_code_Embedded_c_code();

	/**
	 * Returns the meta object for class '{@link promela.c_expr <em>cexpr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>cexpr</em>'.
	 * @see promela.c_expr
	 * @generated
	 */
	EClass getc_expr();

	/**
	 * Returns the meta object for the attribute '{@link promela.c_expr#getEmbedded_c_code <em>Embedded ccode</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Embedded ccode</em>'.
	 * @see promela.c_expr#getEmbedded_c_code()
	 * @see #getc_expr()
	 * @generated
	 */
	EAttribute getc_expr_Embedded_c_code();

	/**
	 * Returns the meta object for class '{@link promela.c_decl <em>cdecl</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>cdecl</em>'.
	 * @see promela.c_decl
	 * @generated
	 */
	EClass getc_decl();

	/**
	 * Returns the meta object for the attribute '{@link promela.c_decl#getEmbedded_c_code <em>Embedded ccode</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Embedded ccode</em>'.
	 * @see promela.c_decl#getEmbedded_c_code()
	 * @see #getc_decl()
	 * @generated
	 */
	EAttribute getc_decl_Embedded_c_code();

	/**
	 * Returns the meta object for class '{@link promela.c_track <em>ctrack</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>ctrack</em>'.
	 * @see promela.c_track
	 * @generated
	 */
	EClass getc_track();

	/**
	 * Returns the meta object for the attribute '{@link promela.c_track#getEmbedded_c_code <em>Embedded ccode</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Embedded ccode</em>'.
	 * @see promela.c_track#getEmbedded_c_code()
	 * @see #getc_track()
	 * @generated
	 */
	EAttribute getc_track_Embedded_c_code();

	/**
	 * Returns the meta object for class '{@link promela.c_state <em>cstate</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>cstate</em>'.
	 * @see promela.c_state
	 * @generated
	 */
	EClass getc_state();

	/**
	 * Returns the meta object for the attribute '{@link promela.c_state#getEmbedded_c_code <em>Embedded ccode</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Embedded ccode</em>'.
	 * @see promela.c_state#getEmbedded_c_code()
	 * @see #getc_state()
	 * @generated
	 */
	EAttribute getc_state_Embedded_c_code();

	/**
	 * Returns the meta object for class '{@link promela.options <em>options</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>options</em>'.
	 * @see promela.options
	 * @generated
	 */
	EClass getoptions();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.options#getSequence <em>Sequence</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Sequence</em>'.
	 * @see promela.options#getSequence()
	 * @see #getoptions()
	 * @generated
	 */
	EReference getoptions_Sequence();

	/**
	 * Returns the meta object for class '{@link promela.any_expr <em>any expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>any expr</em>'.
	 * @see promela.any_expr
	 * @generated
	 */
	EClass getany_expr();

	/**
	 * Returns the meta object for class '{@link promela.bin_expr <em>bin expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>bin expr</em>'.
	 * @see promela.bin_expr
	 * @generated
	 */
	EClass getbin_expr();

	/**
	 * Returns the meta object for the attribute '{@link promela.bin_expr#getOperator <em>Operator</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Operator</em>'.
	 * @see promela.bin_expr#getOperator()
	 * @see #getbin_expr()
	 * @generated
	 */
	EAttribute getbin_expr_Operator();

	/**
	 * Returns the meta object for the containment reference '{@link promela.bin_expr#getOperand1 <em>Operand1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand1</em>'.
	 * @see promela.bin_expr#getOperand1()
	 * @see #getbin_expr()
	 * @generated
	 */
	EReference getbin_expr_Operand1();

	/**
	 * Returns the meta object for the containment reference '{@link promela.bin_expr#getOperand2 <em>Operand2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand2</em>'.
	 * @see promela.bin_expr#getOperand2()
	 * @see #getbin_expr()
	 * @generated
	 */
	EReference getbin_expr_Operand2();

	/**
	 * Returns the meta object for class '{@link promela.un_expr <em>un expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>un expr</em>'.
	 * @see promela.un_expr
	 * @generated
	 */
	EClass getun_expr();

	/**
	 * Returns the meta object for the attribute '{@link promela.un_expr#getOperator <em>Operator</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Operator</em>'.
	 * @see promela.un_expr#getOperator()
	 * @see #getun_expr()
	 * @generated
	 */
	EAttribute getun_expr_Operator();

	/**
	 * Returns the meta object for the containment reference '{@link promela.un_expr#getOperand <em>Operand</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand</em>'.
	 * @see promela.un_expr#getOperand()
	 * @see #getun_expr()
	 * @generated
	 */
	EReference getun_expr_Operand();

	/**
	 * Returns the meta object for class '{@link promela.cond_expr <em>cond expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>cond expr</em>'.
	 * @see promela.cond_expr
	 * @generated
	 */
	EClass getcond_expr();

	/**
	 * Returns the meta object for the containment reference '{@link promela.cond_expr#getExpr1 <em>Expr1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Expr1</em>'.
	 * @see promela.cond_expr#getExpr1()
	 * @see #getcond_expr()
	 * @generated
	 */
	EReference getcond_expr_Expr1();

	/**
	 * Returns the meta object for the containment reference '{@link promela.cond_expr#getExpr2 <em>Expr2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Expr2</em>'.
	 * @see promela.cond_expr#getExpr2()
	 * @see #getcond_expr()
	 * @generated
	 */
	EReference getcond_expr_Expr2();

	/**
	 * Returns the meta object for the containment reference '{@link promela.cond_expr#getExpr3 <em>Expr3</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Expr3</em>'.
	 * @see promela.cond_expr#getExpr3()
	 * @see #getcond_expr()
	 * @generated
	 */
	EReference getcond_expr_Expr3();

	/**
	 * Returns the meta object for class '{@link promela.len_expr <em>len expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>len expr</em>'.
	 * @see promela.len_expr
	 * @generated
	 */
	EClass getlen_expr();

	/**
	 * Returns the meta object for the reference '{@link promela.len_expr#getVarref <em>Varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Varref</em>'.
	 * @see promela.len_expr#getVarref()
	 * @see #getlen_expr()
	 * @generated
	 */
	EReference getlen_expr_Varref();

	/**
	 * Returns the meta object for class '{@link promela.timeout_expr <em>timeout expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>timeout expr</em>'.
	 * @see promela.timeout_expr
	 * @generated
	 */
	EClass gettimeout_expr();

	/**
	 * Returns the meta object for class '{@link promela.np__expr <em>np expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>np expr</em>'.
	 * @see promela.np__expr
	 * @generated
	 */
	EClass getnp__expr();

	/**
	 * Returns the meta object for class '{@link promela.enabled_expr <em>enabled expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>enabled expr</em>'.
	 * @see promela.enabled_expr
	 * @generated
	 */
	EClass getenabled_expr();

	/**
	 * Returns the meta object for the containment reference '{@link promela.enabled_expr#getAny_expr <em>Any expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Any expr</em>'.
	 * @see promela.enabled_expr#getAny_expr()
	 * @see #getenabled_expr()
	 * @generated
	 */
	EReference getenabled_expr_Any_expr();

	/**
	 * Returns the meta object for class '{@link promela.pc_value_expr <em>pc value expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>pc value expr</em>'.
	 * @see promela.pc_value_expr
	 * @generated
	 */
	EClass getpc_value_expr();

	/**
	 * Returns the meta object for the containment reference '{@link promela.pc_value_expr#getAny_expr <em>Any expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Any expr</em>'.
	 * @see promela.pc_value_expr#getAny_expr()
	 * @see #getpc_value_expr()
	 * @generated
	 */
	EReference getpc_value_expr_Any_expr();

	/**
	 * Returns the meta object for class '{@link promela.name_expr <em>name expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>name expr</em>'.
	 * @see promela.name_expr
	 * @generated
	 */
	EClass getname_expr();

	/**
	 * Returns the meta object for the attribute '{@link promela.name_expr#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see promela.name_expr#getName()
	 * @see #getname_expr()
	 * @generated
	 */
	EAttribute getname_expr_Name();

	/**
	 * Returns the meta object for the containment reference '{@link promela.name_expr#getAny_expr <em>Any expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Any expr</em>'.
	 * @see promela.name_expr#getAny_expr()
	 * @see #getname_expr()
	 * @generated
	 */
	EReference getname_expr_Any_expr();

	/**
	 * Returns the meta object for the attribute '{@link promela.name_expr#getAt_name <em>At name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>At name</em>'.
	 * @see promela.name_expr#getAt_name()
	 * @see #getname_expr()
	 * @generated
	 */
	EAttribute getname_expr_At_name();

	/**
	 * Returns the meta object for class '{@link promela.run_expr <em>run expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>run expr</em>'.
	 * @see promela.run_expr
	 * @generated
	 */
	EClass getrun_expr();

	/**
	 * Returns the meta object for the reference '{@link promela.run_expr#getProc <em>Proc</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Proc</em>'.
	 * @see promela.run_expr#getProc()
	 * @see #getrun_expr()
	 * @generated
	 */
	EReference getrun_expr_Proc();

	/**
	 * Returns the meta object for the containment reference list '{@link promela.run_expr#getArg_lst <em>Arg lst</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Arg lst</em>'.
	 * @see promela.run_expr#getArg_lst()
	 * @see #getrun_expr()
	 * @generated
	 */
	EReference getrun_expr_Arg_lst();

	/**
	 * Returns the meta object for the attribute '{@link promela.run_expr#getPriority <em>Priority</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Priority</em>'.
	 * @see promela.run_expr#getPriority()
	 * @see #getrun_expr()
	 * @generated
	 */
	EAttribute getrun_expr_Priority();

	/**
	 * Returns the meta object for class '{@link promela.expr <em>expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>expr</em>'.
	 * @see promela.expr
	 * @generated
	 */
	EClass getexpr();

	/**
	 * Returns the meta object for class '{@link promela.andor_expr <em>andor expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>andor expr</em>'.
	 * @see promela.andor_expr
	 * @generated
	 */
	EClass getandor_expr();

	/**
	 * Returns the meta object for the attribute '{@link promela.andor_expr#getOperator <em>Operator</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Operator</em>'.
	 * @see promela.andor_expr#getOperator()
	 * @see #getandor_expr()
	 * @generated
	 */
	EAttribute getandor_expr_Operator();

	/**
	 * Returns the meta object for the containment reference '{@link promela.andor_expr#getOperand1 <em>Operand1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand1</em>'.
	 * @see promela.andor_expr#getOperand1()
	 * @see #getandor_expr()
	 * @generated
	 */
	EReference getandor_expr_Operand1();

	/**
	 * Returns the meta object for the containment reference '{@link promela.andor_expr#getOperand2 <em>Operand2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand2</em>'.
	 * @see promela.andor_expr#getOperand2()
	 * @see #getandor_expr()
	 * @generated
	 */
	EReference getandor_expr_Operand2();

	/**
	 * Returns the meta object for class '{@link promela.chanpoll_expr <em>chanpoll expr</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>chanpoll expr</em>'.
	 * @see promela.chanpoll_expr
	 * @generated
	 */
	EClass getchanpoll_expr();

	/**
	 * Returns the meta object for the attribute '{@link promela.chanpoll_expr#getChan_poll <em>Chan poll</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Chan poll</em>'.
	 * @see promela.chanpoll_expr#getChan_poll()
	 * @see #getchanpoll_expr()
	 * @generated
	 */
	EAttribute getchanpoll_expr_Chan_poll();

	/**
	 * Returns the meta object for the containment reference '{@link promela.chanpoll_expr#getVarref <em>Varref</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Varref</em>'.
	 * @see promela.chanpoll_expr#getVarref()
	 * @see #getchanpoll_expr()
	 * @generated
	 */
	EReference getchanpoll_expr_Varref();

	/**
	 * Returns the meta object for class '{@link promela.constant <em>constant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>constant</em>'.
	 * @see promela.constant
	 * @generated
	 */
	EClass getconstant();

	/**
	 * Returns the meta object for class '{@link promela.num_const <em>num const</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>num const</em>'.
	 * @see promela.num_const
	 * @generated
	 */
	EClass getnum_const();

	/**
	 * Returns the meta object for the attribute '{@link promela.num_const#getNumber <em>Number</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Number</em>'.
	 * @see promela.num_const#getNumber()
	 * @see #getnum_const()
	 * @generated
	 */
	EAttribute getnum_const_Number();

	/**
	 * Returns the meta object for class '{@link promela.enum_const <em>enum const</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>enum const</em>'.
	 * @see promela.enum_const
	 * @generated
	 */
	EClass getenum_const();

	/**
	 * Returns the meta object for the attribute '{@link promela.enum_const#getConst_enum <em>Const enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Const enum</em>'.
	 * @see promela.enum_const#getConst_enum()
	 * @see #getenum_const()
	 * @generated
	 */
	EAttribute getenum_const_Const_enum();

	/**
	 * Returns the meta object for class '{@link promela.mtype_const <em>mtype const</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>mtype const</em>'.
	 * @see promela.mtype_const
	 * @generated
	 */
	EClass getmtype_const();

	/**
	 * Returns the meta object for the reference '{@link promela.mtype_const#getMtype <em>Mtype</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Mtype</em>'.
	 * @see promela.mtype_const#getMtype()
	 * @see #getmtype_const()
	 * @generated
	 */
	EReference getmtype_const_Mtype();

	/**
	 * Returns the meta object for enum '{@link promela.visibleenum <em>visibleenum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>visibleenum</em>'.
	 * @see promela.visibleenum
	 * @generated
	 */
	EEnum getvisibleenum();

	/**
	 * Returns the meta object for enum '{@link promela.primitivetypeenum <em>primitivetypeenum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>primitivetypeenum</em>'.
	 * @see promela.primitivetypeenum
	 * @generated
	 */
	EEnum getprimitivetypeenum();

	/**
	 * Returns the meta object for enum '{@link promela.channelassertion <em>channelassertion</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>channelassertion</em>'.
	 * @see promela.channelassertion
	 * @generated
	 */
	EEnum getchannelassertion();

	/**
	 * Returns the meta object for enum '{@link promela.sendtypeenum <em>sendtypeenum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>sendtypeenum</em>'.
	 * @see promela.sendtypeenum
	 * @generated
	 */
	EEnum getsendtypeenum();

	/**
	 * Returns the meta object for enum '{@link promela.receivetypeenum <em>receivetypeenum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>receivetypeenum</em>'.
	 * @see promela.receivetypeenum
	 * @generated
	 */
	EEnum getreceivetypeenum();

	/**
	 * Returns the meta object for enum '{@link promela.polltypeenum <em>polltypeenum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>polltypeenum</em>'.
	 * @see promela.polltypeenum
	 * @generated
	 */
	EEnum getpolltypeenum();

	/**
	 * Returns the meta object for enum '{@link promela.andor <em>andor</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>andor</em>'.
	 * @see promela.andor
	 * @generated
	 */
	EEnum getandor();

	/**
	 * Returns the meta object for enum '{@link promela.binarop <em>binarop</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>binarop</em>'.
	 * @see promela.binarop
	 * @generated
	 */
	EEnum getbinarop();

	/**
	 * Returns the meta object for enum '{@link promela.unarop <em>unarop</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>unarop</em>'.
	 * @see promela.unarop
	 * @generated
	 */
	EEnum getunarop();

	/**
	 * Returns the meta object for enum '{@link promela.chanpoll <em>chanpoll</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>chanpoll</em>'.
	 * @see promela.chanpoll
	 * @generated
	 */
	EEnum getchanpoll();

	/**
	 * Returns the meta object for enum '{@link promela.constenum <em>constenum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>constenum</em>'.
	 * @see promela.constenum
	 * @generated
	 */
	EEnum getconstenum();

	/**
	 * Returns the factory that creates the instances of the model.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the factory that creates the instances of the model.
	 * @generated
	 */
	PromelaFactory getPromelaFactory();

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
		 * The meta object literal for the '{@link promela.impl.specImpl <em>spec</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.specImpl
		 * @see promela.impl.PromelaPackageImpl#getspec()
		 * @generated
		 */
		EClass SPEC = eINSTANCE.getspec();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SPEC__NAME = eINSTANCE.getspec_Name();

		/**
		 * The meta object literal for the '<em><b>Utypes</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SPEC__UTYPES = eINSTANCE.getspec_Utypes();

		/**
		 * The meta object literal for the '<em><b>Mtypes</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SPEC__MTYPES = eINSTANCE.getspec_Mtypes();

		/**
		 * The meta object literal for the '<em><b>One decls</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SPEC__ONE_DECLS = eINSTANCE.getspec_One_decls();

		/**
		 * The meta object literal for the '<em><b>Proctypes</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SPEC__PROCTYPES = eINSTANCE.getspec_Proctypes();

		/**
		 * The meta object literal for the '<em><b>Inits</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SPEC__INITS = eINSTANCE.getspec_Inits();

		/**
		 * The meta object literal for the '<em><b>Nevers</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SPEC__NEVERS = eINSTANCE.getspec_Nevers();

		/**
		 * The meta object literal for the '<em><b>Traces</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SPEC__TRACES = eINSTANCE.getspec_Traces();

		/**
		 * The meta object literal for the '<em><b>Primitivetypes</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SPEC__PRIMITIVETYPES = eINSTANCE.getspec_Primitivetypes();

		/**
		 * The meta object literal for the '{@link promela.impl.proctypeImpl <em>proctype</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.proctypeImpl
		 * @see promela.impl.PromelaPackageImpl#getproctype()
		 * @generated
		 */
		EClass PROCTYPE = eINSTANCE.getproctype();

		/**
		 * The meta object literal for the '<em><b>Active</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PROCTYPE__ACTIVE = eINSTANCE.getproctype_Active();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PROCTYPE__NAME = eINSTANCE.getproctype_Name();

		/**
		 * The meta object literal for the '<em><b>One decls</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCTYPE__ONE_DECLS = eINSTANCE.getproctype_One_decls();

		/**
		 * The meta object literal for the '<em><b>Priority</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PROCTYPE__PRIORITY = eINSTANCE.getproctype_Priority();

		/**
		 * The meta object literal for the '<em><b>Enabler</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCTYPE__ENABLER = eINSTANCE.getproctype_Enabler();

		/**
		 * The meta object literal for the '<em><b>Sequence</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROCTYPE__SEQUENCE = eINSTANCE.getproctype_Sequence();

		/**
		 * The meta object literal for the '{@link promela.impl.initImpl <em>init</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.initImpl
		 * @see promela.impl.PromelaPackageImpl#getinit()
		 * @generated
		 */
		EClass INIT = eINSTANCE.getinit();

		/**
		 * The meta object literal for the '<em><b>Priority</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute INIT__PRIORITY = eINSTANCE.getinit_Priority();

		/**
		 * The meta object literal for the '<em><b>Sequence</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference INIT__SEQUENCE = eINSTANCE.getinit_Sequence();

		/**
		 * The meta object literal for the '{@link promela.impl.neverImpl <em>never</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.neverImpl
		 * @see promela.impl.PromelaPackageImpl#getnever()
		 * @generated
		 */
		EClass NEVER = eINSTANCE.getnever();

		/**
		 * The meta object literal for the '<em><b>Sequence</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference NEVER__SEQUENCE = eINSTANCE.getnever_Sequence();

		/**
		 * The meta object literal for the '{@link promela.impl.traceImpl <em>trace</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.traceImpl
		 * @see promela.impl.PromelaPackageImpl#gettrace()
		 * @generated
		 */
		EClass TRACE = eINSTANCE.gettrace();

		/**
		 * The meta object literal for the '<em><b>Sequence</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference TRACE__SEQUENCE = eINSTANCE.gettrace_Sequence();

		/**
		 * The meta object literal for the '{@link promela.impl.utypeImpl <em>utype</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.utypeImpl
		 * @see promela.impl.PromelaPackageImpl#getutype()
		 * @generated
		 */
		EClass UTYPE = eINSTANCE.getutype();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute UTYPE__NAME = eINSTANCE.getutype_Name();

		/**
		 * The meta object literal for the '<em><b>One decls</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference UTYPE__ONE_DECLS = eINSTANCE.getutype_One_decls();

		/**
		 * The meta object literal for the '{@link promela.impl.mtypeImpl <em>mtype</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.mtypeImpl
		 * @see promela.impl.PromelaPackageImpl#getmtype()
		 * @generated
		 */
		EClass MTYPE = eINSTANCE.getmtype();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MTYPE__NAME = eINSTANCE.getmtype_Name();

		/**
		 * The meta object literal for the '{@link promela.impl.one_declImpl <em>one decl</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.one_declImpl
		 * @see promela.impl.PromelaPackageImpl#getone_decl()
		 * @generated
		 */
		EClass ONE_DECL = eINSTANCE.getone_decl();

		/**
		 * The meta object literal for the '<em><b>Visible</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ONE_DECL__VISIBLE = eINSTANCE.getone_decl_Visible();

		/**
		 * The meta object literal for the '<em><b>Typename</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ONE_DECL__TYPENAME = eINSTANCE.getone_decl_Typename();

		/**
		 * The meta object literal for the '<em><b>Ivars</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ONE_DECL__IVARS = eINSTANCE.getone_decl_Ivars();

		/**
		 * The meta object literal for the '{@link promela.impl.typenameImpl <em>typename</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.typenameImpl
		 * @see promela.impl.PromelaPackageImpl#gettypename()
		 * @generated
		 */
		EClass TYPENAME = eINSTANCE.gettypename();

		/**
		 * The meta object literal for the '{@link promela.impl.primitivetypeImpl <em>primitivetype</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.primitivetypeImpl
		 * @see promela.impl.PromelaPackageImpl#getprimitivetype()
		 * @generated
		 */
		EClass PRIMITIVETYPE = eINSTANCE.getprimitivetype();

		/**
		 * The meta object literal for the '<em><b>Primitivetype</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PRIMITIVETYPE__PRIMITIVETYPE = eINSTANCE.getprimitivetype_Primitivetype();

		/**
		 * The meta object literal for the '{@link promela.impl.unameImpl <em>uname</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.unameImpl
		 * @see promela.impl.PromelaPackageImpl#getuname()
		 * @generated
		 */
		EClass UNAME = eINSTANCE.getuname();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute UNAME__NAME = eINSTANCE.getuname_Name();

		/**
		 * The meta object literal for the '{@link promela.impl.stepImpl <em>step</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.stepImpl
		 * @see promela.impl.PromelaPackageImpl#getstep()
		 * @generated
		 */
		EClass STEP = eINSTANCE.getstep();

		/**
		 * The meta object literal for the '{@link promela.impl.stmntImpl <em>stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getstmnt()
		 * @generated
		 */
		EClass STMNT = eINSTANCE.getstmnt();

		/**
		 * The meta object literal for the '<em><b>Label</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute STMNT__LABEL = eINSTANCE.getstmnt_Label();

		/**
		 * The meta object literal for the '<em><b>Unless</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference STMNT__UNLESS = eINSTANCE.getstmnt_Unless();

		/**
		 * The meta object literal for the '{@link promela.impl.chanassertImpl <em>chanassert</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.chanassertImpl
		 * @see promela.impl.PromelaPackageImpl#getchanassert()
		 * @generated
		 */
		EClass CHANASSERT = eINSTANCE.getchanassert();

		/**
		 * The meta object literal for the '<em><b>Channel assertion</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CHANASSERT__CHANNEL_ASSERTION = eINSTANCE.getchanassert_Channel_assertion();

		/**
		 * The meta object literal for the '<em><b>Varrefs</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CHANASSERT__VARREFS = eINSTANCE.getchanassert_Varrefs();

		/**
		 * The meta object literal for the '{@link promela.impl.ivarImpl <em>ivar</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.ivarImpl
		 * @see promela.impl.PromelaPackageImpl#getivar()
		 * @generated
		 */
		EClass IVAR = eINSTANCE.getivar();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute IVAR__NAME = eINSTANCE.getivar_Name();

		/**
		 * The meta object literal for the '<em><b>Constant</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute IVAR__CONSTANT = eINSTANCE.getivar_Constant();

		/**
		 * The meta object literal for the '<em><b>Any expr or ch init</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference IVAR__ANY_EXPR_OR_CH_INIT = eINSTANCE.getivar_Any_expr_or_ch_init();

		/**
		 * The meta object literal for the '{@link promela.impl.any_expr_or_ch_initImpl <em>any expr or ch init</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.any_expr_or_ch_initImpl
		 * @see promela.impl.PromelaPackageImpl#getany_expr_or_ch_init()
		 * @generated
		 */
		EClass ANY_EXPR_OR_CH_INIT = eINSTANCE.getany_expr_or_ch_init();

		/**
		 * The meta object literal for the '{@link promela.impl.ch_initImpl <em>ch init</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.ch_initImpl
		 * @see promela.impl.PromelaPackageImpl#getch_init()
		 * @generated
		 */
		EClass CH_INIT = eINSTANCE.getch_init();

		/**
		 * The meta object literal for the '<em><b>Constant</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CH_INIT__CONSTANT = eINSTANCE.getch_init_Constant();

		/**
		 * The meta object literal for the '<em><b>Typenames</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CH_INIT__TYPENAMES = eINSTANCE.getch_init_Typenames();

		/**
		 * The meta object literal for the '{@link promela.impl.varrefImpl <em>varref</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.varrefImpl
		 * @see promela.impl.PromelaPackageImpl#getvarref()
		 * @generated
		 */
		EClass VARREF = eINSTANCE.getvarref();

		/**
		 * The meta object literal for the '<em><b>Var name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute VARREF__VAR_NAME = eINSTANCE.getvarref_Var_name();

		/**
		 * The meta object literal for the '<em><b>Any expr</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference VARREF__ANY_EXPR = eINSTANCE.getvarref_Any_expr();

		/**
		 * The meta object literal for the '<em><b>Varref</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference VARREF__VARREF = eINSTANCE.getvarref_Varref();

		/**
		 * The meta object literal for the '{@link promela.impl.sendImpl <em>send</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.sendImpl
		 * @see promela.impl.PromelaPackageImpl#getsend()
		 * @generated
		 */
		EClass SEND = eINSTANCE.getsend();

		/**
		 * The meta object literal for the '<em><b>Varref</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SEND__VARREF = eINSTANCE.getsend_Varref();

		/**
		 * The meta object literal for the '<em><b>Send args</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SEND__SEND_ARGS = eINSTANCE.getsend_Send_args();

		/**
		 * The meta object literal for the '<em><b>Sendtype</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SEND__SENDTYPE = eINSTANCE.getsend_Sendtype();

		/**
		 * The meta object literal for the '{@link promela.impl.receiveImpl <em>receive</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.receiveImpl
		 * @see promela.impl.PromelaPackageImpl#getreceive()
		 * @generated
		 */
		EClass RECEIVE = eINSTANCE.getreceive();

		/**
		 * The meta object literal for the '<em><b>Varref</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference RECEIVE__VARREF = eINSTANCE.getreceive_Varref();

		/**
		 * The meta object literal for the '<em><b>Recv args</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference RECEIVE__RECV_ARGS = eINSTANCE.getreceive_Recv_args();

		/**
		 * The meta object literal for the '<em><b>Receivetype</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute RECEIVE__RECEIVETYPE = eINSTANCE.getreceive_Receivetype();

		/**
		 * The meta object literal for the '{@link promela.impl.pollImpl <em>poll</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.pollImpl
		 * @see promela.impl.PromelaPackageImpl#getpoll()
		 * @generated
		 */
		EClass POLL = eINSTANCE.getpoll();

		/**
		 * The meta object literal for the '<em><b>Varref</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference POLL__VARREF = eINSTANCE.getpoll_Varref();

		/**
		 * The meta object literal for the '<em><b>Recv args</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference POLL__RECV_ARGS = eINSTANCE.getpoll_Recv_args();

		/**
		 * The meta object literal for the '<em><b>Polltype</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute POLL__POLLTYPE = eINSTANCE.getpoll_Polltype();

		/**
		 * The meta object literal for the '{@link promela.impl.recv_argImpl <em>recv arg</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.recv_argImpl
		 * @see promela.impl.PromelaPackageImpl#getrecv_arg()
		 * @generated
		 */
		EClass RECV_ARG = eINSTANCE.getrecv_arg();

		/**
		 * The meta object literal for the '{@link promela.impl.recv_arg_varrefImpl <em>recv arg varref</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.recv_arg_varrefImpl
		 * @see promela.impl.PromelaPackageImpl#getrecv_arg_varref()
		 * @generated
		 */
		EClass RECV_ARG_VARREF = eINSTANCE.getrecv_arg_varref();

		/**
		 * The meta object literal for the '<em><b>Varref</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference RECV_ARG_VARREF__VARREF = eINSTANCE.getrecv_arg_varref_Varref();

		/**
		 * The meta object literal for the '{@link promela.impl.recv_arg_eval_exprImpl <em>recv arg eval expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.recv_arg_eval_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getrecv_arg_eval_expr()
		 * @generated
		 */
		EClass RECV_ARG_EVAL_EXPR = eINSTANCE.getrecv_arg_eval_expr();

		/**
		 * The meta object literal for the '<em><b>Expr</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference RECV_ARG_EVAL_EXPR__EXPR = eINSTANCE.getrecv_arg_eval_expr_Expr();

		/**
		 * The meta object literal for the '{@link promela.impl.assignImpl <em>assign</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.assignImpl
		 * @see promela.impl.PromelaPackageImpl#getassign()
		 * @generated
		 */
		EClass ASSIGN = eINSTANCE.getassign();

		/**
		 * The meta object literal for the '{@link promela.impl.assign_stdImpl <em>assign std</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.assign_stdImpl
		 * @see promela.impl.PromelaPackageImpl#getassign_std()
		 * @generated
		 */
		EClass ASSIGN_STD = eINSTANCE.getassign_std();

		/**
		 * The meta object literal for the '<em><b>Varref</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ASSIGN_STD__VARREF = eINSTANCE.getassign_std_Varref();

		/**
		 * The meta object literal for the '<em><b>Any expr</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ASSIGN_STD__ANY_EXPR = eINSTANCE.getassign_std_Any_expr();

		/**
		 * The meta object literal for the '{@link promela.impl.assign_incImpl <em>assign inc</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.assign_incImpl
		 * @see promela.impl.PromelaPackageImpl#getassign_inc()
		 * @generated
		 */
		EClass ASSIGN_INC = eINSTANCE.getassign_inc();

		/**
		 * The meta object literal for the '<em><b>Varref</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ASSIGN_INC__VARREF = eINSTANCE.getassign_inc_Varref();

		/**
		 * The meta object literal for the '{@link promela.impl.assign_decImpl <em>assign dec</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.assign_decImpl
		 * @see promela.impl.PromelaPackageImpl#getassign_dec()
		 * @generated
		 */
		EClass ASSIGN_DEC = eINSTANCE.getassign_dec();

		/**
		 * The meta object literal for the '<em><b>Varref</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ASSIGN_DEC__VARREF = eINSTANCE.getassign_dec_Varref();

		/**
		 * The meta object literal for the '{@link promela.impl.if_stmntImpl <em>if stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.if_stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getif_stmnt()
		 * @generated
		 */
		EClass IF_STMNT = eINSTANCE.getif_stmnt();

		/**
		 * The meta object literal for the '<em><b>Options</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference IF_STMNT__OPTIONS = eINSTANCE.getif_stmnt_Options();

		/**
		 * The meta object literal for the '{@link promela.impl.do_stmntImpl <em>do stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.do_stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getdo_stmnt()
		 * @generated
		 */
		EClass DO_STMNT = eINSTANCE.getdo_stmnt();

		/**
		 * The meta object literal for the '<em><b>Options</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DO_STMNT__OPTIONS = eINSTANCE.getdo_stmnt_Options();

		/**
		 * The meta object literal for the '{@link promela.impl.atomic_stmntImpl <em>atomic stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.atomic_stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getatomic_stmnt()
		 * @generated
		 */
		EClass ATOMIC_STMNT = eINSTANCE.getatomic_stmnt();

		/**
		 * The meta object literal for the '<em><b>Sequence</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ATOMIC_STMNT__SEQUENCE = eINSTANCE.getatomic_stmnt_Sequence();

		/**
		 * The meta object literal for the '{@link promela.impl.d_step_stmntImpl <em>dstep stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.d_step_stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getd_step_stmnt()
		 * @generated
		 */
		EClass DSTEP_STMNT = eINSTANCE.getd_step_stmnt();

		/**
		 * The meta object literal for the '<em><b>Sequence</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference DSTEP_STMNT__SEQUENCE = eINSTANCE.getd_step_stmnt_Sequence();

		/**
		 * The meta object literal for the '{@link promela.impl.block_stmntImpl <em>block stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.block_stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getblock_stmnt()
		 * @generated
		 */
		EClass BLOCK_STMNT = eINSTANCE.getblock_stmnt();

		/**
		 * The meta object literal for the '<em><b>Sequence</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BLOCK_STMNT__SEQUENCE = eINSTANCE.getblock_stmnt_Sequence();

		/**
		 * The meta object literal for the '{@link promela.impl.else_stmntImpl <em>else stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.else_stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getelse_stmnt()
		 * @generated
		 */
		EClass ELSE_STMNT = eINSTANCE.getelse_stmnt();

		/**
		 * The meta object literal for the '{@link promela.impl.break_stmntImpl <em>break stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.break_stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getbreak_stmnt()
		 * @generated
		 */
		EClass BREAK_STMNT = eINSTANCE.getbreak_stmnt();

		/**
		 * The meta object literal for the '{@link promela.impl.goto_stmntImpl <em>goto stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.goto_stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getgoto_stmnt()
		 * @generated
		 */
		EClass GOTO_STMNT = eINSTANCE.getgoto_stmnt();

		/**
		 * The meta object literal for the '<em><b>Stmnt</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference GOTO_STMNT__STMNT = eINSTANCE.getgoto_stmnt_Stmnt();

		/**
		 * The meta object literal for the '{@link promela.impl.print_stmntImpl <em>print stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.print_stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getprint_stmnt()
		 * @generated
		 */
		EClass PRINT_STMNT = eINSTANCE.getprint_stmnt();

		/**
		 * The meta object literal for the '<em><b>String</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PRINT_STMNT__STRING = eINSTANCE.getprint_stmnt_String();

		/**
		 * The meta object literal for the '<em><b>Arg lst</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PRINT_STMNT__ARG_LST = eINSTANCE.getprint_stmnt_Arg_lst();

		/**
		 * The meta object literal for the '{@link promela.impl.assert_stmntImpl <em>assert stmnt</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.assert_stmntImpl
		 * @see promela.impl.PromelaPackageImpl#getassert_stmnt()
		 * @generated
		 */
		EClass ASSERT_STMNT = eINSTANCE.getassert_stmnt();

		/**
		 * The meta object literal for the '<em><b>Expr</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ASSERT_STMNT__EXPR = eINSTANCE.getassert_stmnt_Expr();

		/**
		 * The meta object literal for the '{@link promela.impl.c_codeImpl <em>ccode</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.c_codeImpl
		 * @see promela.impl.PromelaPackageImpl#getc_code()
		 * @generated
		 */
		EClass CCODE = eINSTANCE.getc_code();

		/**
		 * The meta object literal for the '<em><b>Embedded ccode</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CCODE__EMBEDDED_CCODE = eINSTANCE.getc_code_Embedded_c_code();

		/**
		 * The meta object literal for the '{@link promela.impl.c_exprImpl <em>cexpr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.c_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getc_expr()
		 * @generated
		 */
		EClass CEXPR = eINSTANCE.getc_expr();

		/**
		 * The meta object literal for the '<em><b>Embedded ccode</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CEXPR__EMBEDDED_CCODE = eINSTANCE.getc_expr_Embedded_c_code();

		/**
		 * The meta object literal for the '{@link promela.impl.c_declImpl <em>cdecl</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.c_declImpl
		 * @see promela.impl.PromelaPackageImpl#getc_decl()
		 * @generated
		 */
		EClass CDECL = eINSTANCE.getc_decl();

		/**
		 * The meta object literal for the '<em><b>Embedded ccode</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CDECL__EMBEDDED_CCODE = eINSTANCE.getc_decl_Embedded_c_code();

		/**
		 * The meta object literal for the '{@link promela.impl.c_trackImpl <em>ctrack</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.c_trackImpl
		 * @see promela.impl.PromelaPackageImpl#getc_track()
		 * @generated
		 */
		EClass CTRACK = eINSTANCE.getc_track();

		/**
		 * The meta object literal for the '<em><b>Embedded ccode</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CTRACK__EMBEDDED_CCODE = eINSTANCE.getc_track_Embedded_c_code();

		/**
		 * The meta object literal for the '{@link promela.impl.c_stateImpl <em>cstate</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.c_stateImpl
		 * @see promela.impl.PromelaPackageImpl#getc_state()
		 * @generated
		 */
		EClass CSTATE = eINSTANCE.getc_state();

		/**
		 * The meta object literal for the '<em><b>Embedded ccode</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CSTATE__EMBEDDED_CCODE = eINSTANCE.getc_state_Embedded_c_code();

		/**
		 * The meta object literal for the '{@link promela.impl.optionsImpl <em>options</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.optionsImpl
		 * @see promela.impl.PromelaPackageImpl#getoptions()
		 * @generated
		 */
		EClass OPTIONS = eINSTANCE.getoptions();

		/**
		 * The meta object literal for the '<em><b>Sequence</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference OPTIONS__SEQUENCE = eINSTANCE.getoptions_Sequence();

		/**
		 * The meta object literal for the '{@link promela.impl.any_exprImpl <em>any expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.any_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getany_expr()
		 * @generated
		 */
		EClass ANY_EXPR = eINSTANCE.getany_expr();

		/**
		 * The meta object literal for the '{@link promela.impl.bin_exprImpl <em>bin expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.bin_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getbin_expr()
		 * @generated
		 */
		EClass BIN_EXPR = eINSTANCE.getbin_expr();

		/**
		 * The meta object literal for the '<em><b>Operator</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BIN_EXPR__OPERATOR = eINSTANCE.getbin_expr_Operator();

		/**
		 * The meta object literal for the '<em><b>Operand1</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BIN_EXPR__OPERAND1 = eINSTANCE.getbin_expr_Operand1();

		/**
		 * The meta object literal for the '<em><b>Operand2</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BIN_EXPR__OPERAND2 = eINSTANCE.getbin_expr_Operand2();

		/**
		 * The meta object literal for the '{@link promela.impl.un_exprImpl <em>un expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.un_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getun_expr()
		 * @generated
		 */
		EClass UN_EXPR = eINSTANCE.getun_expr();

		/**
		 * The meta object literal for the '<em><b>Operator</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute UN_EXPR__OPERATOR = eINSTANCE.getun_expr_Operator();

		/**
		 * The meta object literal for the '<em><b>Operand</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference UN_EXPR__OPERAND = eINSTANCE.getun_expr_Operand();

		/**
		 * The meta object literal for the '{@link promela.impl.cond_exprImpl <em>cond expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.cond_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getcond_expr()
		 * @generated
		 */
		EClass COND_EXPR = eINSTANCE.getcond_expr();

		/**
		 * The meta object literal for the '<em><b>Expr1</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COND_EXPR__EXPR1 = eINSTANCE.getcond_expr_Expr1();

		/**
		 * The meta object literal for the '<em><b>Expr2</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COND_EXPR__EXPR2 = eINSTANCE.getcond_expr_Expr2();

		/**
		 * The meta object literal for the '<em><b>Expr3</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COND_EXPR__EXPR3 = eINSTANCE.getcond_expr_Expr3();

		/**
		 * The meta object literal for the '{@link promela.impl.len_exprImpl <em>len expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.len_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getlen_expr()
		 * @generated
		 */
		EClass LEN_EXPR = eINSTANCE.getlen_expr();

		/**
		 * The meta object literal for the '<em><b>Varref</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference LEN_EXPR__VARREF = eINSTANCE.getlen_expr_Varref();

		/**
		 * The meta object literal for the '{@link promela.impl.timeout_exprImpl <em>timeout expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.timeout_exprImpl
		 * @see promela.impl.PromelaPackageImpl#gettimeout_expr()
		 * @generated
		 */
		EClass TIMEOUT_EXPR = eINSTANCE.gettimeout_expr();

		/**
		 * The meta object literal for the '{@link promela.impl.np__exprImpl <em>np expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.np__exprImpl
		 * @see promela.impl.PromelaPackageImpl#getnp__expr()
		 * @generated
		 */
		EClass NP_EXPR = eINSTANCE.getnp__expr();

		/**
		 * The meta object literal for the '{@link promela.impl.enabled_exprImpl <em>enabled expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.enabled_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getenabled_expr()
		 * @generated
		 */
		EClass ENABLED_EXPR = eINSTANCE.getenabled_expr();

		/**
		 * The meta object literal for the '<em><b>Any expr</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ENABLED_EXPR__ANY_EXPR = eINSTANCE.getenabled_expr_Any_expr();

		/**
		 * The meta object literal for the '{@link promela.impl.pc_value_exprImpl <em>pc value expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.pc_value_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getpc_value_expr()
		 * @generated
		 */
		EClass PC_VALUE_EXPR = eINSTANCE.getpc_value_expr();

		/**
		 * The meta object literal for the '<em><b>Any expr</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PC_VALUE_EXPR__ANY_EXPR = eINSTANCE.getpc_value_expr_Any_expr();

		/**
		 * The meta object literal for the '{@link promela.impl.name_exprImpl <em>name expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.name_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getname_expr()
		 * @generated
		 */
		EClass NAME_EXPR = eINSTANCE.getname_expr();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute NAME_EXPR__NAME = eINSTANCE.getname_expr_Name();

		/**
		 * The meta object literal for the '<em><b>Any expr</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference NAME_EXPR__ANY_EXPR = eINSTANCE.getname_expr_Any_expr();

		/**
		 * The meta object literal for the '<em><b>At name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute NAME_EXPR__AT_NAME = eINSTANCE.getname_expr_At_name();

		/**
		 * The meta object literal for the '{@link promela.impl.run_exprImpl <em>run expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.run_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getrun_expr()
		 * @generated
		 */
		EClass RUN_EXPR = eINSTANCE.getrun_expr();

		/**
		 * The meta object literal for the '<em><b>Proc</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference RUN_EXPR__PROC = eINSTANCE.getrun_expr_Proc();

		/**
		 * The meta object literal for the '<em><b>Arg lst</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference RUN_EXPR__ARG_LST = eINSTANCE.getrun_expr_Arg_lst();

		/**
		 * The meta object literal for the '<em><b>Priority</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute RUN_EXPR__PRIORITY = eINSTANCE.getrun_expr_Priority();

		/**
		 * The meta object literal for the '{@link promela.impl.exprImpl <em>expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.exprImpl
		 * @see promela.impl.PromelaPackageImpl#getexpr()
		 * @generated
		 */
		EClass EXPR = eINSTANCE.getexpr();

		/**
		 * The meta object literal for the '{@link promela.impl.andor_exprImpl <em>andor expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.andor_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getandor_expr()
		 * @generated
		 */
		EClass ANDOR_EXPR = eINSTANCE.getandor_expr();

		/**
		 * The meta object literal for the '<em><b>Operator</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ANDOR_EXPR__OPERATOR = eINSTANCE.getandor_expr_Operator();

		/**
		 * The meta object literal for the '<em><b>Operand1</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ANDOR_EXPR__OPERAND1 = eINSTANCE.getandor_expr_Operand1();

		/**
		 * The meta object literal for the '<em><b>Operand2</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ANDOR_EXPR__OPERAND2 = eINSTANCE.getandor_expr_Operand2();

		/**
		 * The meta object literal for the '{@link promela.impl.chanpoll_exprImpl <em>chanpoll expr</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.chanpoll_exprImpl
		 * @see promela.impl.PromelaPackageImpl#getchanpoll_expr()
		 * @generated
		 */
		EClass CHANPOLL_EXPR = eINSTANCE.getchanpoll_expr();

		/**
		 * The meta object literal for the '<em><b>Chan poll</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CHANPOLL_EXPR__CHAN_POLL = eINSTANCE.getchanpoll_expr_Chan_poll();

		/**
		 * The meta object literal for the '<em><b>Varref</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CHANPOLL_EXPR__VARREF = eINSTANCE.getchanpoll_expr_Varref();

		/**
		 * The meta object literal for the '{@link promela.impl.constantImpl <em>constant</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.constantImpl
		 * @see promela.impl.PromelaPackageImpl#getconstant()
		 * @generated
		 */
		EClass CONSTANT = eINSTANCE.getconstant();

		/**
		 * The meta object literal for the '{@link promela.impl.num_constImpl <em>num const</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.num_constImpl
		 * @see promela.impl.PromelaPackageImpl#getnum_const()
		 * @generated
		 */
		EClass NUM_CONST = eINSTANCE.getnum_const();

		/**
		 * The meta object literal for the '<em><b>Number</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute NUM_CONST__NUMBER = eINSTANCE.getnum_const_Number();

		/**
		 * The meta object literal for the '{@link promela.impl.enum_constImpl <em>enum const</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.enum_constImpl
		 * @see promela.impl.PromelaPackageImpl#getenum_const()
		 * @generated
		 */
		EClass ENUM_CONST = eINSTANCE.getenum_const();

		/**
		 * The meta object literal for the '<em><b>Const enum</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ENUM_CONST__CONST_ENUM = eINSTANCE.getenum_const_Const_enum();

		/**
		 * The meta object literal for the '{@link promela.impl.mtype_constImpl <em>mtype const</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.impl.mtype_constImpl
		 * @see promela.impl.PromelaPackageImpl#getmtype_const()
		 * @generated
		 */
		EClass MTYPE_CONST = eINSTANCE.getmtype_const();

		/**
		 * The meta object literal for the '<em><b>Mtype</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MTYPE_CONST__MTYPE = eINSTANCE.getmtype_const_Mtype();

		/**
		 * The meta object literal for the '{@link promela.visibleenum <em>visibleenum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.visibleenum
		 * @see promela.impl.PromelaPackageImpl#getvisibleenum()
		 * @generated
		 */
		EEnum VISIBLEENUM = eINSTANCE.getvisibleenum();

		/**
		 * The meta object literal for the '{@link promela.primitivetypeenum <em>primitivetypeenum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.primitivetypeenum
		 * @see promela.impl.PromelaPackageImpl#getprimitivetypeenum()
		 * @generated
		 */
		EEnum PRIMITIVETYPEENUM = eINSTANCE.getprimitivetypeenum();

		/**
		 * The meta object literal for the '{@link promela.channelassertion <em>channelassertion</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.channelassertion
		 * @see promela.impl.PromelaPackageImpl#getchannelassertion()
		 * @generated
		 */
		EEnum CHANNELASSERTION = eINSTANCE.getchannelassertion();

		/**
		 * The meta object literal for the '{@link promela.sendtypeenum <em>sendtypeenum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.sendtypeenum
		 * @see promela.impl.PromelaPackageImpl#getsendtypeenum()
		 * @generated
		 */
		EEnum SENDTYPEENUM = eINSTANCE.getsendtypeenum();

		/**
		 * The meta object literal for the '{@link promela.receivetypeenum <em>receivetypeenum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.receivetypeenum
		 * @see promela.impl.PromelaPackageImpl#getreceivetypeenum()
		 * @generated
		 */
		EEnum RECEIVETYPEENUM = eINSTANCE.getreceivetypeenum();

		/**
		 * The meta object literal for the '{@link promela.polltypeenum <em>polltypeenum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.polltypeenum
		 * @see promela.impl.PromelaPackageImpl#getpolltypeenum()
		 * @generated
		 */
		EEnum POLLTYPEENUM = eINSTANCE.getpolltypeenum();

		/**
		 * The meta object literal for the '{@link promela.andor <em>andor</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.andor
		 * @see promela.impl.PromelaPackageImpl#getandor()
		 * @generated
		 */
		EEnum ANDOR = eINSTANCE.getandor();

		/**
		 * The meta object literal for the '{@link promela.binarop <em>binarop</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.binarop
		 * @see promela.impl.PromelaPackageImpl#getbinarop()
		 * @generated
		 */
		EEnum BINAROP = eINSTANCE.getbinarop();

		/**
		 * The meta object literal for the '{@link promela.unarop <em>unarop</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.unarop
		 * @see promela.impl.PromelaPackageImpl#getunarop()
		 * @generated
		 */
		EEnum UNAROP = eINSTANCE.getunarop();

		/**
		 * The meta object literal for the '{@link promela.chanpoll <em>chanpoll</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.chanpoll
		 * @see promela.impl.PromelaPackageImpl#getchanpoll()
		 * @generated
		 */
		EEnum CHANPOLL = eINSTANCE.getchanpoll();

		/**
		 * The meta object literal for the '{@link promela.constenum <em>constenum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see promela.constenum
		 * @see promela.impl.PromelaPackageImpl#getconstenum()
		 * @generated
		 */
		EEnum CONSTENUM = eINSTANCE.getconstenum();

	}

} //PromelaPackage
