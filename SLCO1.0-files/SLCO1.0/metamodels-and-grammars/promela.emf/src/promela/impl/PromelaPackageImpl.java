/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

import org.eclipse.emf.ecore.impl.EPackageImpl;

import promela.PromelaFactory;
import promela.PromelaPackage;
import promela.andor;
import promela.andor_expr;
import promela.any_expr;
import promela.any_expr_or_ch_init;
import promela.assert_stmnt;
import promela.assign;
import promela.assign_dec;
import promela.assign_inc;
import promela.assign_std;
import promela.atomic_stmnt;
import promela.bin_expr;
import promela.binarop;
import promela.block_stmnt;
import promela.break_stmnt;
import promela.c_code;
import promela.c_decl;
import promela.c_expr;
import promela.c_state;
import promela.c_track;
import promela.ch_init;
import promela.chanassert;
import promela.channelassertion;
import promela.chanpoll;
import promela.chanpoll_expr;
import promela.cond_expr;
import promela.constant;
import promela.constenum;
import promela.d_step_stmnt;
import promela.do_stmnt;
import promela.else_stmnt;
import promela.enabled_expr;
import promela.enum_const;
import promela.expr;
import promela.goto_stmnt;
import promela.if_stmnt;
import promela.init;
import promela.ivar;
import promela.len_expr;
import promela.mtype;
import promela.mtype_const;
import promela.name_expr;
import promela.never;
import promela.np__expr;
import promela.num_const;
import promela.one_decl;
import promela.options;
import promela.pc_value_expr;
import promela.poll;
import promela.polltypeenum;
import promela.primitivetype;
import promela.primitivetypeenum;
import promela.print_stmnt;
import promela.proctype;
import promela.receive;
import promela.receivetypeenum;
import promela.recv_arg;
import promela.recv_arg_eval_expr;
import promela.recv_arg_varref;
import promela.run_expr;
import promela.send;
import promela.sendtypeenum;
import promela.spec;
import promela.step;
import promela.stmnt;
import promela.timeout_expr;
import promela.trace;
import promela.typename;
import promela.un_expr;
import promela.uname;
import promela.unarop;
import promela.utype;
import promela.varref;
import promela.visibleenum;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Package</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class PromelaPackageImpl extends EPackageImpl implements PromelaPackage {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass specEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass proctypeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass initEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass neverEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass traceEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass utypeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass mtypeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass one_declEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass typenameEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass primitivetypeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass unameEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass stepEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass chanassertEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass ivarEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass any_expr_or_ch_initEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass ch_initEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass varrefEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass sendEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass receiveEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass pollEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass recv_argEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass recv_arg_varrefEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass recv_arg_eval_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass assignEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass assign_stdEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass assign_incEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass assign_decEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass if_stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass do_stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass atomic_stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass d_step_stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass block_stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass else_stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass break_stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass goto_stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass print_stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass assert_stmntEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass c_codeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass c_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass c_declEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass c_trackEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass c_stateEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass optionsEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass any_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass bin_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass un_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass cond_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass len_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass timeout_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass np__exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass enabled_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass pc_value_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass name_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass run_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass andor_exprEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass chanpoll_exprEClass = null;

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
	private EClass num_constEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass enum_constEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass mtype_constEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum visibleenumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum primitivetypeenumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum channelassertionEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum sendtypeenumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum receivetypeenumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum polltypeenumEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum andorEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum binaropEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum unaropEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum chanpollEEnum = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum constenumEEnum = null;

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
	 * @see promela.PromelaPackage#eNS_URI
	 * @see #init()
	 * @generated
	 */
	private PromelaPackageImpl() {
		super(eNS_URI, PromelaFactory.eINSTANCE);
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
	 * <p>This method is used to initialize {@link PromelaPackage#eINSTANCE} when that field is accessed.
	 * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #eNS_URI
	 * @see #createPackageContents()
	 * @see #initializePackageContents()
	 * @generated
	 */
	public static PromelaPackage init() {
		if (isInited) return (PromelaPackage)EPackage.Registry.INSTANCE.getEPackage(PromelaPackage.eNS_URI);

		// Obtain or create and register package
		PromelaPackageImpl thePromelaPackage = (PromelaPackageImpl)(EPackage.Registry.INSTANCE.get(eNS_URI) instanceof PromelaPackageImpl ? EPackage.Registry.INSTANCE.get(eNS_URI) : new PromelaPackageImpl());

		isInited = true;

		// Create package meta-data objects
		thePromelaPackage.createPackageContents();

		// Initialize created meta-data
		thePromelaPackage.initializePackageContents();

		// Mark meta-data to indicate it can't be changed
		thePromelaPackage.freeze();

  
		// Update the registry and return the package
		EPackage.Registry.INSTANCE.put(PromelaPackage.eNS_URI, thePromelaPackage);
		return thePromelaPackage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getspec() {
		return specEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getspec_Name() {
		return (EAttribute)specEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getspec_Utypes() {
		return (EReference)specEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getspec_Mtypes() {
		return (EReference)specEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getspec_One_decls() {
		return (EReference)specEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getspec_Proctypes() {
		return (EReference)specEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getspec_Inits() {
		return (EReference)specEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getspec_Nevers() {
		return (EReference)specEClass.getEStructuralFeatures().get(6);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getspec_Traces() {
		return (EReference)specEClass.getEStructuralFeatures().get(7);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getspec_Primitivetypes() {
		return (EReference)specEClass.getEStructuralFeatures().get(8);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getproctype() {
		return proctypeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getproctype_Active() {
		return (EAttribute)proctypeEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getproctype_Name() {
		return (EAttribute)proctypeEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getproctype_One_decls() {
		return (EReference)proctypeEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getproctype_Priority() {
		return (EAttribute)proctypeEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getproctype_Enabler() {
		return (EReference)proctypeEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getproctype_Sequence() {
		return (EReference)proctypeEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getinit() {
		return initEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getinit_Priority() {
		return (EAttribute)initEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getinit_Sequence() {
		return (EReference)initEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getnever() {
		return neverEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getnever_Sequence() {
		return (EReference)neverEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass gettrace() {
		return traceEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference gettrace_Sequence() {
		return (EReference)traceEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getutype() {
		return utypeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getutype_Name() {
		return (EAttribute)utypeEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getutype_One_decls() {
		return (EReference)utypeEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getmtype() {
		return mtypeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getmtype_Name() {
		return (EAttribute)mtypeEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getone_decl() {
		return one_declEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getone_decl_Visible() {
		return (EAttribute)one_declEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getone_decl_Typename() {
		return (EReference)one_declEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getone_decl_Ivars() {
		return (EReference)one_declEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass gettypename() {
		return typenameEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getprimitivetype() {
		return primitivetypeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getprimitivetype_Primitivetype() {
		return (EAttribute)primitivetypeEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getuname() {
		return unameEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getuname_Name() {
		return (EAttribute)unameEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getstep() {
		return stepEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getstmnt() {
		return stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getstmnt_Label() {
		return (EAttribute)stmntEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getstmnt_Unless() {
		return (EReference)stmntEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getchanassert() {
		return chanassertEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getchanassert_Channel_assertion() {
		return (EAttribute)chanassertEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getchanassert_Varrefs() {
		return (EReference)chanassertEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getivar() {
		return ivarEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getivar_Name() {
		return (EAttribute)ivarEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getivar_Constant() {
		return (EAttribute)ivarEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getivar_Any_expr_or_ch_init() {
		return (EReference)ivarEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getany_expr_or_ch_init() {
		return any_expr_or_ch_initEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getch_init() {
		return ch_initEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getch_init_Constant() {
		return (EAttribute)ch_initEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getch_init_Typenames() {
		return (EReference)ch_initEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getvarref() {
		return varrefEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getvarref_Var_name() {
		return (EAttribute)varrefEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getvarref_Any_expr() {
		return (EReference)varrefEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getvarref_Varref() {
		return (EReference)varrefEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getsend() {
		return sendEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getsend_Varref() {
		return (EReference)sendEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getsend_Send_args() {
		return (EReference)sendEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getsend_Sendtype() {
		return (EAttribute)sendEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getreceive() {
		return receiveEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getreceive_Varref() {
		return (EReference)receiveEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getreceive_Recv_args() {
		return (EReference)receiveEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getreceive_Receivetype() {
		return (EAttribute)receiveEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getpoll() {
		return pollEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getpoll_Varref() {
		return (EReference)pollEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getpoll_Recv_args() {
		return (EReference)pollEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getpoll_Polltype() {
		return (EAttribute)pollEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getrecv_arg() {
		return recv_argEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getrecv_arg_varref() {
		return recv_arg_varrefEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getrecv_arg_varref_Varref() {
		return (EReference)recv_arg_varrefEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getrecv_arg_eval_expr() {
		return recv_arg_eval_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getrecv_arg_eval_expr_Expr() {
		return (EReference)recv_arg_eval_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getassign() {
		return assignEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getassign_std() {
		return assign_stdEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getassign_std_Varref() {
		return (EReference)assign_stdEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getassign_std_Any_expr() {
		return (EReference)assign_stdEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getassign_inc() {
		return assign_incEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getassign_inc_Varref() {
		return (EReference)assign_incEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getassign_dec() {
		return assign_decEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getassign_dec_Varref() {
		return (EReference)assign_decEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getif_stmnt() {
		return if_stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getif_stmnt_Options() {
		return (EReference)if_stmntEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getdo_stmnt() {
		return do_stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getdo_stmnt_Options() {
		return (EReference)do_stmntEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getatomic_stmnt() {
		return atomic_stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getatomic_stmnt_Sequence() {
		return (EReference)atomic_stmntEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getd_step_stmnt() {
		return d_step_stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getd_step_stmnt_Sequence() {
		return (EReference)d_step_stmntEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getblock_stmnt() {
		return block_stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getblock_stmnt_Sequence() {
		return (EReference)block_stmntEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getelse_stmnt() {
		return else_stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getbreak_stmnt() {
		return break_stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getgoto_stmnt() {
		return goto_stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getgoto_stmnt_Stmnt() {
		return (EReference)goto_stmntEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getprint_stmnt() {
		return print_stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getprint_stmnt_String() {
		return (EAttribute)print_stmntEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getprint_stmnt_Arg_lst() {
		return (EReference)print_stmntEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getassert_stmnt() {
		return assert_stmntEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getassert_stmnt_Expr() {
		return (EReference)assert_stmntEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getc_code() {
		return c_codeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getc_code_Embedded_c_code() {
		return (EAttribute)c_codeEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getc_expr() {
		return c_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getc_expr_Embedded_c_code() {
		return (EAttribute)c_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getc_decl() {
		return c_declEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getc_decl_Embedded_c_code() {
		return (EAttribute)c_declEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getc_track() {
		return c_trackEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getc_track_Embedded_c_code() {
		return (EAttribute)c_trackEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getc_state() {
		return c_stateEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getc_state_Embedded_c_code() {
		return (EAttribute)c_stateEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getoptions() {
		return optionsEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getoptions_Sequence() {
		return (EReference)optionsEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getany_expr() {
		return any_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getbin_expr() {
		return bin_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getbin_expr_Operator() {
		return (EAttribute)bin_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getbin_expr_Operand1() {
		return (EReference)bin_exprEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getbin_expr_Operand2() {
		return (EReference)bin_exprEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getun_expr() {
		return un_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getun_expr_Operator() {
		return (EAttribute)un_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getun_expr_Operand() {
		return (EReference)un_exprEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getcond_expr() {
		return cond_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getcond_expr_Expr1() {
		return (EReference)cond_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getcond_expr_Expr2() {
		return (EReference)cond_exprEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getcond_expr_Expr3() {
		return (EReference)cond_exprEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getlen_expr() {
		return len_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getlen_expr_Varref() {
		return (EReference)len_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass gettimeout_expr() {
		return timeout_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getnp__expr() {
		return np__exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getenabled_expr() {
		return enabled_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getenabled_expr_Any_expr() {
		return (EReference)enabled_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getpc_value_expr() {
		return pc_value_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getpc_value_expr_Any_expr() {
		return (EReference)pc_value_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getname_expr() {
		return name_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getname_expr_Name() {
		return (EAttribute)name_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getname_expr_Any_expr() {
		return (EReference)name_exprEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getname_expr_At_name() {
		return (EAttribute)name_exprEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getrun_expr() {
		return run_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getrun_expr_Proc() {
		return (EReference)run_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getrun_expr_Arg_lst() {
		return (EReference)run_exprEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getrun_expr_Priority() {
		return (EAttribute)run_exprEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getexpr() {
		return exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getandor_expr() {
		return andor_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getandor_expr_Operator() {
		return (EAttribute)andor_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getandor_expr_Operand1() {
		return (EReference)andor_exprEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getandor_expr_Operand2() {
		return (EReference)andor_exprEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getchanpoll_expr() {
		return chanpoll_exprEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getchanpoll_expr_Chan_poll() {
		return (EAttribute)chanpoll_exprEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getchanpoll_expr_Varref() {
		return (EReference)chanpoll_exprEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getconstant() {
		return constantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getnum_const() {
		return num_constEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getnum_const_Number() {
		return (EAttribute)num_constEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getenum_const() {
		return enum_constEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getenum_const_Const_enum() {
		return (EAttribute)enum_constEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getmtype_const() {
		return mtype_constEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getmtype_const_Mtype() {
		return (EReference)mtype_constEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getvisibleenum() {
		return visibleenumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getprimitivetypeenum() {
		return primitivetypeenumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getchannelassertion() {
		return channelassertionEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getsendtypeenum() {
		return sendtypeenumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getreceivetypeenum() {
		return receivetypeenumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getpolltypeenum() {
		return polltypeenumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getandor() {
		return andorEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getbinarop() {
		return binaropEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getunarop() {
		return unaropEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getchanpoll() {
		return chanpollEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getconstenum() {
		return constenumEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PromelaFactory getPromelaFactory() {
		return (PromelaFactory)getEFactoryInstance();
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
		specEClass = createEClass(SPEC);
		createEAttribute(specEClass, SPEC__NAME);
		createEReference(specEClass, SPEC__UTYPES);
		createEReference(specEClass, SPEC__MTYPES);
		createEReference(specEClass, SPEC__ONE_DECLS);
		createEReference(specEClass, SPEC__PROCTYPES);
		createEReference(specEClass, SPEC__INITS);
		createEReference(specEClass, SPEC__NEVERS);
		createEReference(specEClass, SPEC__TRACES);
		createEReference(specEClass, SPEC__PRIMITIVETYPES);

		proctypeEClass = createEClass(PROCTYPE);
		createEAttribute(proctypeEClass, PROCTYPE__ACTIVE);
		createEAttribute(proctypeEClass, PROCTYPE__NAME);
		createEReference(proctypeEClass, PROCTYPE__ONE_DECLS);
		createEAttribute(proctypeEClass, PROCTYPE__PRIORITY);
		createEReference(proctypeEClass, PROCTYPE__ENABLER);
		createEReference(proctypeEClass, PROCTYPE__SEQUENCE);

		initEClass = createEClass(INIT);
		createEAttribute(initEClass, INIT__PRIORITY);
		createEReference(initEClass, INIT__SEQUENCE);

		neverEClass = createEClass(NEVER);
		createEReference(neverEClass, NEVER__SEQUENCE);

		traceEClass = createEClass(TRACE);
		createEReference(traceEClass, TRACE__SEQUENCE);

		utypeEClass = createEClass(UTYPE);
		createEAttribute(utypeEClass, UTYPE__NAME);
		createEReference(utypeEClass, UTYPE__ONE_DECLS);

		mtypeEClass = createEClass(MTYPE);
		createEAttribute(mtypeEClass, MTYPE__NAME);

		one_declEClass = createEClass(ONE_DECL);
		createEAttribute(one_declEClass, ONE_DECL__VISIBLE);
		createEReference(one_declEClass, ONE_DECL__TYPENAME);
		createEReference(one_declEClass, ONE_DECL__IVARS);

		typenameEClass = createEClass(TYPENAME);

		primitivetypeEClass = createEClass(PRIMITIVETYPE);
		createEAttribute(primitivetypeEClass, PRIMITIVETYPE__PRIMITIVETYPE);

		unameEClass = createEClass(UNAME);
		createEAttribute(unameEClass, UNAME__NAME);

		stepEClass = createEClass(STEP);

		stmntEClass = createEClass(STMNT);
		createEAttribute(stmntEClass, STMNT__LABEL);
		createEReference(stmntEClass, STMNT__UNLESS);

		chanassertEClass = createEClass(CHANASSERT);
		createEAttribute(chanassertEClass, CHANASSERT__CHANNEL_ASSERTION);
		createEReference(chanassertEClass, CHANASSERT__VARREFS);

		ivarEClass = createEClass(IVAR);
		createEAttribute(ivarEClass, IVAR__NAME);
		createEAttribute(ivarEClass, IVAR__CONSTANT);
		createEReference(ivarEClass, IVAR__ANY_EXPR_OR_CH_INIT);

		any_expr_or_ch_initEClass = createEClass(ANY_EXPR_OR_CH_INIT);

		ch_initEClass = createEClass(CH_INIT);
		createEAttribute(ch_initEClass, CH_INIT__CONSTANT);
		createEReference(ch_initEClass, CH_INIT__TYPENAMES);

		varrefEClass = createEClass(VARREF);
		createEAttribute(varrefEClass, VARREF__VAR_NAME);
		createEReference(varrefEClass, VARREF__ANY_EXPR);
		createEReference(varrefEClass, VARREF__VARREF);

		sendEClass = createEClass(SEND);
		createEReference(sendEClass, SEND__VARREF);
		createEReference(sendEClass, SEND__SEND_ARGS);
		createEAttribute(sendEClass, SEND__SENDTYPE);

		receiveEClass = createEClass(RECEIVE);
		createEReference(receiveEClass, RECEIVE__VARREF);
		createEReference(receiveEClass, RECEIVE__RECV_ARGS);
		createEAttribute(receiveEClass, RECEIVE__RECEIVETYPE);

		pollEClass = createEClass(POLL);
		createEReference(pollEClass, POLL__VARREF);
		createEReference(pollEClass, POLL__RECV_ARGS);
		createEAttribute(pollEClass, POLL__POLLTYPE);

		recv_argEClass = createEClass(RECV_ARG);

		recv_arg_varrefEClass = createEClass(RECV_ARG_VARREF);
		createEReference(recv_arg_varrefEClass, RECV_ARG_VARREF__VARREF);

		recv_arg_eval_exprEClass = createEClass(RECV_ARG_EVAL_EXPR);
		createEReference(recv_arg_eval_exprEClass, RECV_ARG_EVAL_EXPR__EXPR);

		assignEClass = createEClass(ASSIGN);

		assign_stdEClass = createEClass(ASSIGN_STD);
		createEReference(assign_stdEClass, ASSIGN_STD__VARREF);
		createEReference(assign_stdEClass, ASSIGN_STD__ANY_EXPR);

		assign_incEClass = createEClass(ASSIGN_INC);
		createEReference(assign_incEClass, ASSIGN_INC__VARREF);

		assign_decEClass = createEClass(ASSIGN_DEC);
		createEReference(assign_decEClass, ASSIGN_DEC__VARREF);

		if_stmntEClass = createEClass(IF_STMNT);
		createEReference(if_stmntEClass, IF_STMNT__OPTIONS);

		do_stmntEClass = createEClass(DO_STMNT);
		createEReference(do_stmntEClass, DO_STMNT__OPTIONS);

		atomic_stmntEClass = createEClass(ATOMIC_STMNT);
		createEReference(atomic_stmntEClass, ATOMIC_STMNT__SEQUENCE);

		d_step_stmntEClass = createEClass(DSTEP_STMNT);
		createEReference(d_step_stmntEClass, DSTEP_STMNT__SEQUENCE);

		block_stmntEClass = createEClass(BLOCK_STMNT);
		createEReference(block_stmntEClass, BLOCK_STMNT__SEQUENCE);

		else_stmntEClass = createEClass(ELSE_STMNT);

		break_stmntEClass = createEClass(BREAK_STMNT);

		goto_stmntEClass = createEClass(GOTO_STMNT);
		createEReference(goto_stmntEClass, GOTO_STMNT__STMNT);

		print_stmntEClass = createEClass(PRINT_STMNT);
		createEAttribute(print_stmntEClass, PRINT_STMNT__STRING);
		createEReference(print_stmntEClass, PRINT_STMNT__ARG_LST);

		assert_stmntEClass = createEClass(ASSERT_STMNT);
		createEReference(assert_stmntEClass, ASSERT_STMNT__EXPR);

		c_codeEClass = createEClass(CCODE);
		createEAttribute(c_codeEClass, CCODE__EMBEDDED_CCODE);

		c_exprEClass = createEClass(CEXPR);
		createEAttribute(c_exprEClass, CEXPR__EMBEDDED_CCODE);

		c_declEClass = createEClass(CDECL);
		createEAttribute(c_declEClass, CDECL__EMBEDDED_CCODE);

		c_trackEClass = createEClass(CTRACK);
		createEAttribute(c_trackEClass, CTRACK__EMBEDDED_CCODE);

		c_stateEClass = createEClass(CSTATE);
		createEAttribute(c_stateEClass, CSTATE__EMBEDDED_CCODE);

		optionsEClass = createEClass(OPTIONS);
		createEReference(optionsEClass, OPTIONS__SEQUENCE);

		any_exprEClass = createEClass(ANY_EXPR);

		bin_exprEClass = createEClass(BIN_EXPR);
		createEAttribute(bin_exprEClass, BIN_EXPR__OPERATOR);
		createEReference(bin_exprEClass, BIN_EXPR__OPERAND1);
		createEReference(bin_exprEClass, BIN_EXPR__OPERAND2);

		un_exprEClass = createEClass(UN_EXPR);
		createEAttribute(un_exprEClass, UN_EXPR__OPERATOR);
		createEReference(un_exprEClass, UN_EXPR__OPERAND);

		cond_exprEClass = createEClass(COND_EXPR);
		createEReference(cond_exprEClass, COND_EXPR__EXPR1);
		createEReference(cond_exprEClass, COND_EXPR__EXPR2);
		createEReference(cond_exprEClass, COND_EXPR__EXPR3);

		len_exprEClass = createEClass(LEN_EXPR);
		createEReference(len_exprEClass, LEN_EXPR__VARREF);

		timeout_exprEClass = createEClass(TIMEOUT_EXPR);

		np__exprEClass = createEClass(NP_EXPR);

		enabled_exprEClass = createEClass(ENABLED_EXPR);
		createEReference(enabled_exprEClass, ENABLED_EXPR__ANY_EXPR);

		pc_value_exprEClass = createEClass(PC_VALUE_EXPR);
		createEReference(pc_value_exprEClass, PC_VALUE_EXPR__ANY_EXPR);

		name_exprEClass = createEClass(NAME_EXPR);
		createEAttribute(name_exprEClass, NAME_EXPR__NAME);
		createEReference(name_exprEClass, NAME_EXPR__ANY_EXPR);
		createEAttribute(name_exprEClass, NAME_EXPR__AT_NAME);

		run_exprEClass = createEClass(RUN_EXPR);
		createEReference(run_exprEClass, RUN_EXPR__PROC);
		createEReference(run_exprEClass, RUN_EXPR__ARG_LST);
		createEAttribute(run_exprEClass, RUN_EXPR__PRIORITY);

		exprEClass = createEClass(EXPR);

		andor_exprEClass = createEClass(ANDOR_EXPR);
		createEAttribute(andor_exprEClass, ANDOR_EXPR__OPERATOR);
		createEReference(andor_exprEClass, ANDOR_EXPR__OPERAND1);
		createEReference(andor_exprEClass, ANDOR_EXPR__OPERAND2);

		chanpoll_exprEClass = createEClass(CHANPOLL_EXPR);
		createEAttribute(chanpoll_exprEClass, CHANPOLL_EXPR__CHAN_POLL);
		createEReference(chanpoll_exprEClass, CHANPOLL_EXPR__VARREF);

		constantEClass = createEClass(CONSTANT);

		num_constEClass = createEClass(NUM_CONST);
		createEAttribute(num_constEClass, NUM_CONST__NUMBER);

		enum_constEClass = createEClass(ENUM_CONST);
		createEAttribute(enum_constEClass, ENUM_CONST__CONST_ENUM);

		mtype_constEClass = createEClass(MTYPE_CONST);
		createEReference(mtype_constEClass, MTYPE_CONST__MTYPE);

		// Create enums
		visibleenumEEnum = createEEnum(VISIBLEENUM);
		primitivetypeenumEEnum = createEEnum(PRIMITIVETYPEENUM);
		channelassertionEEnum = createEEnum(CHANNELASSERTION);
		sendtypeenumEEnum = createEEnum(SENDTYPEENUM);
		receivetypeenumEEnum = createEEnum(RECEIVETYPEENUM);
		polltypeenumEEnum = createEEnum(POLLTYPEENUM);
		andorEEnum = createEEnum(ANDOR);
		binaropEEnum = createEEnum(BINAROP);
		unaropEEnum = createEEnum(UNAROP);
		chanpollEEnum = createEEnum(CHANPOLL);
		constenumEEnum = createEEnum(CONSTENUM);
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
		one_declEClass.getESuperTypes().add(this.getstep());
		primitivetypeEClass.getESuperTypes().add(this.gettypename());
		unameEClass.getESuperTypes().add(this.gettypename());
		stmntEClass.getESuperTypes().add(this.getstep());
		chanassertEClass.getESuperTypes().add(this.getstep());
		ch_initEClass.getESuperTypes().add(this.getany_expr_or_ch_init());
		varrefEClass.getESuperTypes().add(this.getany_expr());
		sendEClass.getESuperTypes().add(this.getstmnt());
		receiveEClass.getESuperTypes().add(this.getstmnt());
		pollEClass.getESuperTypes().add(this.getany_expr());
		recv_arg_varrefEClass.getESuperTypes().add(this.getrecv_arg());
		recv_arg_eval_exprEClass.getESuperTypes().add(this.getrecv_arg());
		assignEClass.getESuperTypes().add(this.getstmnt());
		assign_stdEClass.getESuperTypes().add(this.getassign());
		assign_incEClass.getESuperTypes().add(this.getassign());
		assign_decEClass.getESuperTypes().add(this.getassign());
		if_stmntEClass.getESuperTypes().add(this.getstmnt());
		do_stmntEClass.getESuperTypes().add(this.getstmnt());
		atomic_stmntEClass.getESuperTypes().add(this.getstmnt());
		d_step_stmntEClass.getESuperTypes().add(this.getstmnt());
		block_stmntEClass.getESuperTypes().add(this.getstmnt());
		else_stmntEClass.getESuperTypes().add(this.getstmnt());
		break_stmntEClass.getESuperTypes().add(this.getstmnt());
		goto_stmntEClass.getESuperTypes().add(this.getstmnt());
		print_stmntEClass.getESuperTypes().add(this.getstmnt());
		assert_stmntEClass.getESuperTypes().add(this.getstmnt());
		c_codeEClass.getESuperTypes().add(this.getstmnt());
		c_exprEClass.getESuperTypes().add(this.getstmnt());
		c_declEClass.getESuperTypes().add(this.getstmnt());
		c_trackEClass.getESuperTypes().add(this.getstmnt());
		c_stateEClass.getESuperTypes().add(this.getstmnt());
		any_exprEClass.getESuperTypes().add(this.getexpr());
		any_exprEClass.getESuperTypes().add(this.getany_expr_or_ch_init());
		bin_exprEClass.getESuperTypes().add(this.getany_expr());
		un_exprEClass.getESuperTypes().add(this.getany_expr());
		cond_exprEClass.getESuperTypes().add(this.getany_expr());
		len_exprEClass.getESuperTypes().add(this.getany_expr());
		timeout_exprEClass.getESuperTypes().add(this.getany_expr());
		np__exprEClass.getESuperTypes().add(this.getany_expr());
		enabled_exprEClass.getESuperTypes().add(this.getany_expr());
		pc_value_exprEClass.getESuperTypes().add(this.getany_expr());
		name_exprEClass.getESuperTypes().add(this.getany_expr());
		run_exprEClass.getESuperTypes().add(this.getany_expr());
		exprEClass.getESuperTypes().add(this.getstmnt());
		andor_exprEClass.getESuperTypes().add(this.getexpr());
		chanpoll_exprEClass.getESuperTypes().add(this.getexpr());
		constantEClass.getESuperTypes().add(this.getany_expr());
		num_constEClass.getESuperTypes().add(this.getconstant());
		num_constEClass.getESuperTypes().add(this.getrecv_arg());
		enum_constEClass.getESuperTypes().add(this.getconstant());
		mtype_constEClass.getESuperTypes().add(this.getconstant());
		mtype_constEClass.getESuperTypes().add(this.getrecv_arg());

		// Initialize classes and features; add operations and parameters
		initEClass(specEClass, spec.class, "spec", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getspec_Name(), ecorePackage.getEString(), "name", null, 1, 1, spec.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getspec_Utypes(), this.getutype(), null, "utypes", null, 0, -1, spec.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getspec_Mtypes(), this.getmtype(), null, "mtypes", null, 0, -1, spec.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getspec_One_decls(), this.getone_decl(), null, "one_decls", null, 0, -1, spec.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getspec_Proctypes(), this.getproctype(), null, "proctypes", null, 0, -1, spec.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getspec_Inits(), this.getinit(), null, "inits", null, 0, -1, spec.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getspec_Nevers(), this.getnever(), null, "nevers", null, 0, -1, spec.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getspec_Traces(), this.gettrace(), null, "traces", null, 0, -1, spec.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getspec_Primitivetypes(), this.getprimitivetype(), null, "primitivetypes", null, 0, -1, spec.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(proctypeEClass, proctype.class, "proctype", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getproctype_Active(), ecorePackage.getEInt(), "active", null, 0, 1, proctype.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getproctype_Name(), ecorePackage.getEString(), "name", null, 1, 1, proctype.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getproctype_One_decls(), this.getone_decl(), null, "one_decls", null, 0, -1, proctype.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getproctype_Priority(), ecorePackage.getEInt(), "priority", null, 0, 1, proctype.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getproctype_Enabler(), this.getexpr(), null, "enabler", null, 0, 1, proctype.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getproctype_Sequence(), this.getstep(), null, "sequence", null, 1, -1, proctype.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(initEClass, init.class, "init", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getinit_Priority(), ecorePackage.getEInt(), "priority", null, 0, 1, init.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getinit_Sequence(), this.getstep(), null, "sequence", null, 1, -1, init.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(neverEClass, never.class, "never", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getnever_Sequence(), this.getstep(), null, "sequence", null, 1, -1, never.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(traceEClass, trace.class, "trace", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(gettrace_Sequence(), this.getstep(), null, "sequence", null, 1, -1, trace.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(utypeEClass, utype.class, "utype", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getutype_Name(), ecorePackage.getEString(), "name", null, 1, 1, utype.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getutype_One_decls(), this.getone_decl(), null, "one_decls", null, 1, -1, utype.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(mtypeEClass, mtype.class, "mtype", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getmtype_Name(), ecorePackage.getEString(), "name", null, 1, 1, mtype.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(one_declEClass, one_decl.class, "one_decl", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getone_decl_Visible(), this.getvisibleenum(), "visible", null, 0, 1, one_decl.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getone_decl_Typename(), this.gettypename(), null, "typename", null, 1, 1, one_decl.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getone_decl_Ivars(), this.getivar(), null, "ivars", null, 1, -1, one_decl.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(typenameEClass, typename.class, "typename", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(primitivetypeEClass, primitivetype.class, "primitivetype", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getprimitivetype_Primitivetype(), this.getprimitivetypeenum(), "primitivetype", null, 1, 1, primitivetype.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(unameEClass, uname.class, "uname", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getuname_Name(), ecorePackage.getEString(), "name", null, 1, 1, uname.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(stepEClass, step.class, "step", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(stmntEClass, stmnt.class, "stmnt", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getstmnt_Label(), ecorePackage.getEString(), "label", null, 0, 1, stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getstmnt_Unless(), this.getstmnt(), null, "unless", null, 0, 1, stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(chanassertEClass, chanassert.class, "chanassert", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getchanassert_Channel_assertion(), this.getchannelassertion(), "channel_assertion", null, 1, 1, chanassert.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getchanassert_Varrefs(), this.getvarref(), null, "varrefs", null, 1, -1, chanassert.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(ivarEClass, ivar.class, "ivar", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getivar_Name(), ecorePackage.getEString(), "name", null, 1, 1, ivar.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getivar_Constant(), ecorePackage.getEInt(), "constant", null, 0, 1, ivar.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getivar_Any_expr_or_ch_init(), this.getany_expr_or_ch_init(), null, "any_expr_or_ch_init", null, 0, 1, ivar.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(any_expr_or_ch_initEClass, any_expr_or_ch_init.class, "any_expr_or_ch_init", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(ch_initEClass, ch_init.class, "ch_init", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getch_init_Constant(), ecorePackage.getEInt(), "constant", null, 0, 1, ch_init.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getch_init_Typenames(), this.gettypename(), null, "typenames", null, 1, -1, ch_init.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(varrefEClass, varref.class, "varref", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getvarref_Var_name(), ecorePackage.getEString(), "var_name", null, 1, 1, varref.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getvarref_Any_expr(), this.getany_expr(), null, "any_expr", null, 0, 1, varref.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getvarref_Varref(), this.getvarref(), null, "varref", null, 0, 1, varref.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(sendEClass, send.class, "send", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getsend_Varref(), this.getvarref(), null, "varref", null, 1, 1, send.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getsend_Send_args(), this.getany_expr(), null, "send_args", null, 1, -1, send.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getsend_Sendtype(), this.getsendtypeenum(), "sendtype", "0", 1, 1, send.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(receiveEClass, receive.class, "receive", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getreceive_Varref(), this.getvarref(), null, "varref", null, 1, 1, receive.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getreceive_Recv_args(), this.getrecv_arg(), null, "recv_args", null, 1, -1, receive.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getreceive_Receivetype(), this.getreceivetypeenum(), "receivetype", "0", 1, 1, receive.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(pollEClass, poll.class, "poll", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getpoll_Varref(), this.getvarref(), null, "varref", null, 1, 1, poll.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getpoll_Recv_args(), this.getrecv_arg(), null, "recv_args", null, 1, -1, poll.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getpoll_Polltype(), this.getpolltypeenum(), "polltype", "0", 1, 1, poll.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(recv_argEClass, recv_arg.class, "recv_arg", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(recv_arg_varrefEClass, recv_arg_varref.class, "recv_arg_varref", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getrecv_arg_varref_Varref(), this.getvarref(), null, "varref", null, 1, 1, recv_arg_varref.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(recv_arg_eval_exprEClass, recv_arg_eval_expr.class, "recv_arg_eval_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getrecv_arg_eval_expr_Expr(), this.getany_expr(), null, "expr", null, 1, 1, recv_arg_eval_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(assignEClass, assign.class, "assign", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(assign_stdEClass, assign_std.class, "assign_std", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getassign_std_Varref(), this.getvarref(), null, "varref", null, 1, 1, assign_std.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getassign_std_Any_expr(), this.getany_expr(), null, "any_expr", null, 1, 1, assign_std.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(assign_incEClass, assign_inc.class, "assign_inc", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getassign_inc_Varref(), this.getvarref(), null, "varref", null, 1, 1, assign_inc.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(assign_decEClass, assign_dec.class, "assign_dec", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getassign_dec_Varref(), this.getvarref(), null, "varref", null, 1, 1, assign_dec.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(if_stmntEClass, if_stmnt.class, "if_stmnt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getif_stmnt_Options(), this.getoptions(), null, "options", null, 1, 1, if_stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(do_stmntEClass, do_stmnt.class, "do_stmnt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getdo_stmnt_Options(), this.getoptions(), null, "options", null, 1, 1, do_stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(atomic_stmntEClass, atomic_stmnt.class, "atomic_stmnt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getatomic_stmnt_Sequence(), this.getstep(), null, "sequence", null, 1, -1, atomic_stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(d_step_stmntEClass, d_step_stmnt.class, "d_step_stmnt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getd_step_stmnt_Sequence(), this.getstep(), null, "sequence", null, 1, -1, d_step_stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(block_stmntEClass, block_stmnt.class, "block_stmnt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getblock_stmnt_Sequence(), this.getstep(), null, "sequence", null, 1, -1, block_stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(else_stmntEClass, else_stmnt.class, "else_stmnt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(break_stmntEClass, break_stmnt.class, "break_stmnt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(goto_stmntEClass, goto_stmnt.class, "goto_stmnt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getgoto_stmnt_Stmnt(), this.getstmnt(), null, "stmnt", null, 1, 1, goto_stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(print_stmntEClass, print_stmnt.class, "print_stmnt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getprint_stmnt_String(), ecorePackage.getEString(), "string", null, 1, 1, print_stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getprint_stmnt_Arg_lst(), this.getany_expr(), null, "arg_lst", null, 0, -1, print_stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(assert_stmntEClass, assert_stmnt.class, "assert_stmnt", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getassert_stmnt_Expr(), this.getexpr(), null, "expr", null, 1, 1, assert_stmnt.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(c_codeEClass, c_code.class, "c_code", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getc_code_Embedded_c_code(), ecorePackage.getEString(), "embedded_c_code", null, 1, 1, c_code.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(c_exprEClass, c_expr.class, "c_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getc_expr_Embedded_c_code(), ecorePackage.getEString(), "embedded_c_code", null, 1, 1, c_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(c_declEClass, c_decl.class, "c_decl", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getc_decl_Embedded_c_code(), ecorePackage.getEString(), "embedded_c_code", null, 1, 1, c_decl.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(c_trackEClass, c_track.class, "c_track", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getc_track_Embedded_c_code(), ecorePackage.getEString(), "embedded_c_code", null, 1, 1, c_track.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(c_stateEClass, c_state.class, "c_state", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getc_state_Embedded_c_code(), ecorePackage.getEString(), "embedded_c_code", null, 1, 1, c_state.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(optionsEClass, options.class, "options", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getoptions_Sequence(), this.getstep(), null, "sequence", null, 1, -1, options.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(any_exprEClass, any_expr.class, "any_expr", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(bin_exprEClass, bin_expr.class, "bin_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getbin_expr_Operator(), this.getbinarop(), "operator", null, 1, 1, bin_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getbin_expr_Operand1(), this.getany_expr(), null, "operand1", null, 1, 1, bin_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getbin_expr_Operand2(), this.getany_expr(), null, "operand2", null, 1, 1, bin_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(un_exprEClass, un_expr.class, "un_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getun_expr_Operator(), this.getunarop(), "operator", null, 1, 1, un_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getun_expr_Operand(), this.getany_expr(), null, "operand", null, 1, 1, un_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(cond_exprEClass, cond_expr.class, "cond_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getcond_expr_Expr1(), this.getany_expr(), null, "expr1", null, 1, 1, cond_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getcond_expr_Expr2(), this.getany_expr(), null, "expr2", null, 1, 1, cond_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getcond_expr_Expr3(), this.getany_expr(), null, "expr3", null, 1, 1, cond_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(len_exprEClass, len_expr.class, "len_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getlen_expr_Varref(), this.getvarref(), null, "varref", null, 1, 1, len_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(timeout_exprEClass, timeout_expr.class, "timeout_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(np__exprEClass, np__expr.class, "np__expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(enabled_exprEClass, enabled_expr.class, "enabled_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getenabled_expr_Any_expr(), this.getany_expr(), null, "any_expr", null, 1, 1, enabled_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(pc_value_exprEClass, pc_value_expr.class, "pc_value_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getpc_value_expr_Any_expr(), this.getany_expr(), null, "any_expr", null, 1, 1, pc_value_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(name_exprEClass, name_expr.class, "name_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getname_expr_Name(), ecorePackage.getEString(), "name", null, 1, 1, name_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getname_expr_Any_expr(), this.getany_expr(), null, "any_expr", null, 0, 1, name_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getname_expr_At_name(), ecorePackage.getEString(), "at_name", null, 1, 1, name_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(run_exprEClass, run_expr.class, "run_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getrun_expr_Proc(), this.getproctype(), null, "proc", null, 1, 1, run_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getrun_expr_Arg_lst(), this.getany_expr(), null, "arg_lst", null, 0, -1, run_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getrun_expr_Priority(), ecorePackage.getEInt(), "priority", null, 0, 1, run_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(exprEClass, expr.class, "expr", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(andor_exprEClass, andor_expr.class, "andor_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getandor_expr_Operator(), this.getandor(), "operator", null, 1, 1, andor_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getandor_expr_Operand1(), this.getexpr(), null, "operand1", null, 1, 1, andor_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getandor_expr_Operand2(), this.getexpr(), null, "operand2", null, 1, 1, andor_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(chanpoll_exprEClass, chanpoll_expr.class, "chanpoll_expr", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getchanpoll_expr_Chan_poll(), this.getchanpoll(), "chan_poll", null, 1, 1, chanpoll_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getchanpoll_expr_Varref(), this.getvarref(), null, "varref", null, 1, 1, chanpoll_expr.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(constantEClass, constant.class, "constant", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(num_constEClass, num_const.class, "num_const", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getnum_const_Number(), ecorePackage.getEInt(), "number", null, 1, 1, num_const.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(enum_constEClass, enum_const.class, "enum_const", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getenum_const_Const_enum(), this.getconstenum(), "const_enum", null, 1, 1, enum_const.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(mtype_constEClass, mtype_const.class, "mtype_const", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getmtype_const_Mtype(), this.getmtype(), null, "mtype", null, 1, 1, mtype_const.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		// Initialize enums and add enum literals
		initEEnum(visibleenumEEnum, visibleenum.class, "visibleenum");
		addEEnumLiteral(visibleenumEEnum, visibleenum.HIDDEN);
		addEEnumLiteral(visibleenumEEnum, visibleenum.SHOW);

		initEEnum(primitivetypeenumEEnum, primitivetypeenum.class, "primitivetypeenum");
		addEEnumLiteral(primitivetypeenumEEnum, primitivetypeenum.BIT);
		addEEnumLiteral(primitivetypeenumEEnum, primitivetypeenum.BOOL);
		addEEnumLiteral(primitivetypeenumEEnum, primitivetypeenum.BYTE);
		addEEnumLiteral(primitivetypeenumEEnum, primitivetypeenum.SHORT);
		addEEnumLiteral(primitivetypeenumEEnum, primitivetypeenum.INT);
		addEEnumLiteral(primitivetypeenumEEnum, primitivetypeenum.MTYPE);
		addEEnumLiteral(primitivetypeenumEEnum, primitivetypeenum.CHAN);

		initEEnum(channelassertionEEnum, channelassertion.class, "channelassertion");
		addEEnumLiteral(channelassertionEEnum, channelassertion.XR);
		addEEnumLiteral(channelassertionEEnum, channelassertion.XS);

		initEEnum(sendtypeenumEEnum, sendtypeenum.class, "sendtypeenum");
		addEEnumLiteral(sendtypeenumEEnum, sendtypeenum.NORMAL_FIFO);
		addEEnumLiteral(sendtypeenumEEnum, sendtypeenum.SORTED);

		initEEnum(receivetypeenumEEnum, receivetypeenum.class, "receivetypeenum");
		addEEnumLiteral(receivetypeenumEEnum, receivetypeenum.NORMAL);
		addEEnumLiteral(receivetypeenumEEnum, receivetypeenum.RANDOM);
		addEEnumLiteral(receivetypeenumEEnum, receivetypeenum.NORMAL_POLL_SIDE_EFFECT);
		addEEnumLiteral(receivetypeenumEEnum, receivetypeenum.RANDOM_POLL_SIDE_EFFECT);

		initEEnum(polltypeenumEEnum, polltypeenum.class, "polltypeenum");
		addEEnumLiteral(polltypeenumEEnum, polltypeenum.NORMAL_POLL_NO_SIDE_EFFECT);
		addEEnumLiteral(polltypeenumEEnum, polltypeenum.RANDOM_POLL_NO_SIDE_EFFECT);

		initEEnum(andorEEnum, andor.class, "andor");
		addEEnumLiteral(andorEEnum, andor.AND);
		addEEnumLiteral(andorEEnum, andor.OR);

		initEEnum(binaropEEnum, binarop.class, "binarop");
		addEEnumLiteral(binaropEEnum, binarop.PLUS);
		addEEnumLiteral(binaropEEnum, binarop.MINUS);
		addEEnumLiteral(binaropEEnum, binarop.TIMES);
		addEEnumLiteral(binaropEEnum, binarop.DIV);
		addEEnumLiteral(binaropEEnum, binarop.MOD);
		addEEnumLiteral(binaropEEnum, binarop.AMPERSAND);
		addEEnumLiteral(binaropEEnum, binarop.CARET);
		addEEnumLiteral(binaropEEnum, binarop.BAR);
		addEEnumLiteral(binaropEEnum, binarop.GREATER);
		addEEnumLiteral(binaropEEnum, binarop.SMALLER);
		addEEnumLiteral(binaropEEnum, binarop.GEQ);
		addEEnumLiteral(binaropEEnum, binarop.LEQ);
		addEEnumLiteral(binaropEEnum, binarop.EQUALS);
		addEEnumLiteral(binaropEEnum, binarop.DIFFERS);
		addEEnumLiteral(binaropEEnum, binarop.SHIFT_RIGHT);
		addEEnumLiteral(binaropEEnum, binarop.SHIFT_LEFT);
		addEEnumLiteral(binaropEEnum, binarop.AND);
		addEEnumLiteral(binaropEEnum, binarop.OR);

		initEEnum(unaropEEnum, unarop.class, "unarop");
		addEEnumLiteral(unaropEEnum, unarop.NOT);
		addEEnumLiteral(unaropEEnum, unarop.NEG);
		addEEnumLiteral(unaropEEnum, unarop.NOT1);

		initEEnum(chanpollEEnum, chanpoll.class, "chanpoll");
		addEEnumLiteral(chanpollEEnum, chanpoll.FULL);
		addEEnumLiteral(chanpollEEnum, chanpoll.EMPTY);
		addEEnumLiteral(chanpollEEnum, chanpoll.NFULL);
		addEEnumLiteral(chanpollEEnum, chanpoll.NEMTPY);

		initEEnum(constenumEEnum, constenum.class, "constenum");
		addEEnumLiteral(constenumEEnum, constenum.TRUE);
		addEEnumLiteral(constenumEEnum, constenum.FALSE);
		addEEnumLiteral(constenumEEnum, constenum.SKIP);

		// Create resource
		createResource(eNS_URI);
	}

} //PromelaPackageImpl
