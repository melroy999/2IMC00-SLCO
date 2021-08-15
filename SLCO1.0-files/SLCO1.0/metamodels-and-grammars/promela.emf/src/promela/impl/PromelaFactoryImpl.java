/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EDataType;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.impl.EFactoryImpl;

import org.eclipse.emf.ecore.plugin.EcorePlugin;

import promela.*;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Factory</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class PromelaFactoryImpl extends EFactoryImpl implements PromelaFactory {
	/**
	 * Creates the default factory implementation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static PromelaFactory init() {
		try {
			PromelaFactory thePromelaFactory = (PromelaFactory)EPackage.Registry.INSTANCE.getEFactory("promela"); 
			if (thePromelaFactory != null) {
				return thePromelaFactory;
			}
		}
		catch (Exception exception) {
			EcorePlugin.INSTANCE.log(exception);
		}
		return new PromelaFactoryImpl();
	}

	/**
	 * Creates an instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PromelaFactoryImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EObject create(EClass eClass) {
		switch (eClass.getClassifierID()) {
			case PromelaPackage.SPEC: return createspec();
			case PromelaPackage.PROCTYPE: return createproctype();
			case PromelaPackage.INIT: return createinit();
			case PromelaPackage.NEVER: return createnever();
			case PromelaPackage.TRACE: return createtrace();
			case PromelaPackage.UTYPE: return createutype();
			case PromelaPackage.MTYPE: return createmtype();
			case PromelaPackage.ONE_DECL: return createone_decl();
			case PromelaPackage.PRIMITIVETYPE: return createprimitivetype();
			case PromelaPackage.UNAME: return createuname();
			case PromelaPackage.CHANASSERT: return createchanassert();
			case PromelaPackage.IVAR: return createivar();
			case PromelaPackage.CH_INIT: return createch_init();
			case PromelaPackage.VARREF: return createvarref();
			case PromelaPackage.SEND: return createsend();
			case PromelaPackage.RECEIVE: return createreceive();
			case PromelaPackage.RECV_ARG_VARREF: return createrecv_arg_varref();
			case PromelaPackage.RECV_ARG_EVAL_EXPR: return createrecv_arg_eval_expr();
			case PromelaPackage.ASSIGN_STD: return createassign_std();
			case PromelaPackage.ASSIGN_INC: return createassign_inc();
			case PromelaPackage.ASSIGN_DEC: return createassign_dec();
			case PromelaPackage.IF_STMNT: return createif_stmnt();
			case PromelaPackage.DO_STMNT: return createdo_stmnt();
			case PromelaPackage.ATOMIC_STMNT: return createatomic_stmnt();
			case PromelaPackage.DSTEP_STMNT: return created_step_stmnt();
			case PromelaPackage.BLOCK_STMNT: return createblock_stmnt();
			case PromelaPackage.ELSE_STMNT: return createelse_stmnt();
			case PromelaPackage.BREAK_STMNT: return createbreak_stmnt();
			case PromelaPackage.GOTO_STMNT: return creategoto_stmnt();
			case PromelaPackage.PRINT_STMNT: return createprint_stmnt();
			case PromelaPackage.ASSERT_STMNT: return createassert_stmnt();
			case PromelaPackage.CCODE: return createc_code();
			case PromelaPackage.CEXPR: return createc_expr();
			case PromelaPackage.CDECL: return createc_decl();
			case PromelaPackage.CTRACK: return createc_track();
			case PromelaPackage.CSTATE: return createc_state();
			case PromelaPackage.OPTIONS: return createoptions();
			case PromelaPackage.BIN_EXPR: return createbin_expr();
			case PromelaPackage.UN_EXPR: return createun_expr();
			case PromelaPackage.COND_EXPR: return createcond_expr();
			case PromelaPackage.LEN_EXPR: return createlen_expr();
			case PromelaPackage.TIMEOUT_EXPR: return createtimeout_expr();
			case PromelaPackage.NP_EXPR: return createnp__expr();
			case PromelaPackage.ENABLED_EXPR: return createenabled_expr();
			case PromelaPackage.PC_VALUE_EXPR: return createpc_value_expr();
			case PromelaPackage.NAME_EXPR: return createname_expr();
			case PromelaPackage.RUN_EXPR: return createrun_expr();
			case PromelaPackage.ANDOR_EXPR: return createandor_expr();
			case PromelaPackage.CHANPOLL_EXPR: return createchanpoll_expr();
			case PromelaPackage.NUM_CONST: return createnum_const();
			case PromelaPackage.ENUM_CONST: return createenum_const();
			case PromelaPackage.MTYPE_CONST: return createmtype_const();
			default:
				throw new IllegalArgumentException("The class '" + eClass.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object createFromString(EDataType eDataType, String initialValue) {
		switch (eDataType.getClassifierID()) {
			case PromelaPackage.VISIBLEENUM:
				return createvisibleenumFromString(eDataType, initialValue);
			case PromelaPackage.PRIMITIVETYPEENUM:
				return createprimitivetypeenumFromString(eDataType, initialValue);
			case PromelaPackage.CHANNELASSERTION:
				return createchannelassertionFromString(eDataType, initialValue);
			case PromelaPackage.SENDTYPEENUM:
				return createsendtypeenumFromString(eDataType, initialValue);
			case PromelaPackage.RECEIVETYPEENUM:
				return createreceivetypeenumFromString(eDataType, initialValue);
			case PromelaPackage.POLLTYPEENUM:
				return createpolltypeenumFromString(eDataType, initialValue);
			case PromelaPackage.ANDOR:
				return createandorFromString(eDataType, initialValue);
			case PromelaPackage.BINAROP:
				return createbinaropFromString(eDataType, initialValue);
			case PromelaPackage.UNAROP:
				return createunaropFromString(eDataType, initialValue);
			case PromelaPackage.CHANPOLL:
				return createchanpollFromString(eDataType, initialValue);
			case PromelaPackage.CONSTENUM:
				return createconstenumFromString(eDataType, initialValue);
			default:
				throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String convertToString(EDataType eDataType, Object instanceValue) {
		switch (eDataType.getClassifierID()) {
			case PromelaPackage.VISIBLEENUM:
				return convertvisibleenumToString(eDataType, instanceValue);
			case PromelaPackage.PRIMITIVETYPEENUM:
				return convertprimitivetypeenumToString(eDataType, instanceValue);
			case PromelaPackage.CHANNELASSERTION:
				return convertchannelassertionToString(eDataType, instanceValue);
			case PromelaPackage.SENDTYPEENUM:
				return convertsendtypeenumToString(eDataType, instanceValue);
			case PromelaPackage.RECEIVETYPEENUM:
				return convertreceivetypeenumToString(eDataType, instanceValue);
			case PromelaPackage.POLLTYPEENUM:
				return convertpolltypeenumToString(eDataType, instanceValue);
			case PromelaPackage.ANDOR:
				return convertandorToString(eDataType, instanceValue);
			case PromelaPackage.BINAROP:
				return convertbinaropToString(eDataType, instanceValue);
			case PromelaPackage.UNAROP:
				return convertunaropToString(eDataType, instanceValue);
			case PromelaPackage.CHANPOLL:
				return convertchanpollToString(eDataType, instanceValue);
			case PromelaPackage.CONSTENUM:
				return convertconstenumToString(eDataType, instanceValue);
			default:
				throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public spec createspec() {
		specImpl spec = new specImpl();
		return spec;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public proctype createproctype() {
		proctypeImpl proctype = new proctypeImpl();
		return proctype;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public init createinit() {
		initImpl init = new initImpl();
		return init;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public never createnever() {
		neverImpl never = new neverImpl();
		return never;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public trace createtrace() {
		traceImpl trace = new traceImpl();
		return trace;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public utype createutype() {
		utypeImpl utype = new utypeImpl();
		return utype;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public mtype createmtype() {
		mtypeImpl mtype = new mtypeImpl();
		return mtype;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public one_decl createone_decl() {
		one_declImpl one_decl = new one_declImpl();
		return one_decl;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public primitivetype createprimitivetype() {
		primitivetypeImpl primitivetype = new primitivetypeImpl();
		return primitivetype;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public uname createuname() {
		unameImpl uname = new unameImpl();
		return uname;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public chanassert createchanassert() {
		chanassertImpl chanassert = new chanassertImpl();
		return chanassert;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ivar createivar() {
		ivarImpl ivar = new ivarImpl();
		return ivar;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ch_init createch_init() {
		ch_initImpl ch_init = new ch_initImpl();
		return ch_init;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public varref createvarref() {
		varrefImpl varref = new varrefImpl();
		return varref;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public send createsend() {
		sendImpl send = new sendImpl();
		return send;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public receive createreceive() {
		receiveImpl receive = new receiveImpl();
		return receive;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public recv_arg_varref createrecv_arg_varref() {
		recv_arg_varrefImpl recv_arg_varref = new recv_arg_varrefImpl();
		return recv_arg_varref;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public recv_arg_eval_expr createrecv_arg_eval_expr() {
		recv_arg_eval_exprImpl recv_arg_eval_expr = new recv_arg_eval_exprImpl();
		return recv_arg_eval_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public assign_std createassign_std() {
		assign_stdImpl assign_std = new assign_stdImpl();
		return assign_std;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public assign_inc createassign_inc() {
		assign_incImpl assign_inc = new assign_incImpl();
		return assign_inc;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public assign_dec createassign_dec() {
		assign_decImpl assign_dec = new assign_decImpl();
		return assign_dec;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public if_stmnt createif_stmnt() {
		if_stmntImpl if_stmnt = new if_stmntImpl();
		return if_stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public do_stmnt createdo_stmnt() {
		do_stmntImpl do_stmnt = new do_stmntImpl();
		return do_stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public atomic_stmnt createatomic_stmnt() {
		atomic_stmntImpl atomic_stmnt = new atomic_stmntImpl();
		return atomic_stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public d_step_stmnt created_step_stmnt() {
		d_step_stmntImpl d_step_stmnt = new d_step_stmntImpl();
		return d_step_stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public block_stmnt createblock_stmnt() {
		block_stmntImpl block_stmnt = new block_stmntImpl();
		return block_stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public else_stmnt createelse_stmnt() {
		else_stmntImpl else_stmnt = new else_stmntImpl();
		return else_stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public break_stmnt createbreak_stmnt() {
		break_stmntImpl break_stmnt = new break_stmntImpl();
		return break_stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public goto_stmnt creategoto_stmnt() {
		goto_stmntImpl goto_stmnt = new goto_stmntImpl();
		return goto_stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public print_stmnt createprint_stmnt() {
		print_stmntImpl print_stmnt = new print_stmntImpl();
		return print_stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public assert_stmnt createassert_stmnt() {
		assert_stmntImpl assert_stmnt = new assert_stmntImpl();
		return assert_stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public c_code createc_code() {
		c_codeImpl c_code = new c_codeImpl();
		return c_code;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public c_expr createc_expr() {
		c_exprImpl c_expr = new c_exprImpl();
		return c_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public c_decl createc_decl() {
		c_declImpl c_decl = new c_declImpl();
		return c_decl;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public c_track createc_track() {
		c_trackImpl c_track = new c_trackImpl();
		return c_track;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public c_state createc_state() {
		c_stateImpl c_state = new c_stateImpl();
		return c_state;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public options createoptions() {
		optionsImpl options = new optionsImpl();
		return options;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public bin_expr createbin_expr() {
		bin_exprImpl bin_expr = new bin_exprImpl();
		return bin_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public un_expr createun_expr() {
		un_exprImpl un_expr = new un_exprImpl();
		return un_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public cond_expr createcond_expr() {
		cond_exprImpl cond_expr = new cond_exprImpl();
		return cond_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public len_expr createlen_expr() {
		len_exprImpl len_expr = new len_exprImpl();
		return len_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public timeout_expr createtimeout_expr() {
		timeout_exprImpl timeout_expr = new timeout_exprImpl();
		return timeout_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public np__expr createnp__expr() {
		np__exprImpl np__expr = new np__exprImpl();
		return np__expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public enabled_expr createenabled_expr() {
		enabled_exprImpl enabled_expr = new enabled_exprImpl();
		return enabled_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public pc_value_expr createpc_value_expr() {
		pc_value_exprImpl pc_value_expr = new pc_value_exprImpl();
		return pc_value_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public name_expr createname_expr() {
		name_exprImpl name_expr = new name_exprImpl();
		return name_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public run_expr createrun_expr() {
		run_exprImpl run_expr = new run_exprImpl();
		return run_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public andor_expr createandor_expr() {
		andor_exprImpl andor_expr = new andor_exprImpl();
		return andor_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public chanpoll_expr createchanpoll_expr() {
		chanpoll_exprImpl chanpoll_expr = new chanpoll_exprImpl();
		return chanpoll_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public num_const createnum_const() {
		num_constImpl num_const = new num_constImpl();
		return num_const;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public enum_const createenum_const() {
		enum_constImpl enum_const = new enum_constImpl();
		return enum_const;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public mtype_const createmtype_const() {
		mtype_constImpl mtype_const = new mtype_constImpl();
		return mtype_const;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public visibleenum createvisibleenumFromString(EDataType eDataType, String initialValue) {
		visibleenum result = visibleenum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertvisibleenumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public primitivetypeenum createprimitivetypeenumFromString(EDataType eDataType, String initialValue) {
		primitivetypeenum result = primitivetypeenum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertprimitivetypeenumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public channelassertion createchannelassertionFromString(EDataType eDataType, String initialValue) {
		channelassertion result = channelassertion.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertchannelassertionToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public sendtypeenum createsendtypeenumFromString(EDataType eDataType, String initialValue) {
		sendtypeenum result = sendtypeenum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertsendtypeenumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public receivetypeenum createreceivetypeenumFromString(EDataType eDataType, String initialValue) {
		receivetypeenum result = receivetypeenum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertreceivetypeenumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public polltypeenum createpolltypeenumFromString(EDataType eDataType, String initialValue) {
		polltypeenum result = polltypeenum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertpolltypeenumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public andor createandorFromString(EDataType eDataType, String initialValue) {
		andor result = andor.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertandorToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public binarop createbinaropFromString(EDataType eDataType, String initialValue) {
		binarop result = binarop.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertbinaropToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public unarop createunaropFromString(EDataType eDataType, String initialValue) {
		unarop result = unarop.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertunaropToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public chanpoll createchanpollFromString(EDataType eDataType, String initialValue) {
		chanpoll result = chanpoll.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertchanpollToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public constenum createconstenumFromString(EDataType eDataType, String initialValue) {
		constenum result = constenum.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertconstenumToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PromelaPackage getPromelaPackage() {
		return (PromelaPackage)getEPackage();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @deprecated
	 * @generated
	 */
	@Deprecated
	public static PromelaPackage getPackage() {
		return PromelaPackage.eINSTANCE;
	}

} //PromelaFactoryImpl
