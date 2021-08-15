/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.util;

import java.util.List;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;

import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.util.Switch;
import promela.*;

/**
 * <!-- begin-user-doc -->
 * The <b>Switch</b> for the model's inheritance hierarchy.
 * It supports the call {@link #doSwitch(EObject) doSwitch(object)}
 * to invoke the <code>caseXXX</code> method for each class of the model,
 * starting with the actual class of the object
 * and proceeding up the inheritance hierarchy
 * until a non-null result is returned,
 * which is the result of the switch.
 * <!-- end-user-doc -->
 * @see promela.PromelaPackage
 * @generated
 */
public class PromelaSwitch<T> extends Switch<T> {
	/**
	 * The cached model package
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static PromelaPackage modelPackage;

	/**
	 * Creates an instance of the switch.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PromelaSwitch() {
		if (modelPackage == null) {
			modelPackage = PromelaPackage.eINSTANCE;
		}
	}

	/**
	 * Checks whether this is a switch for the given package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @parameter ePackage the package in question.
	 * @return whether this is a switch for the given package.
	 * @generated
	 */
	@Override
	protected boolean isSwitchFor(EPackage ePackage) {
		return ePackage == modelPackage;
	}

	/**
	 * Calls <code>caseXXX</code> for each class of the model until one returns a non null result; it yields that result.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the first non-null result returned by a <code>caseXXX</code> call.
	 * @generated
	 */
	@Override
	protected T doSwitch(int classifierID, EObject theEObject) {
		switch (classifierID) {
			case PromelaPackage.SPEC: {
				spec spec = (spec)theEObject;
				T result = casespec(spec);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.PROCTYPE: {
				proctype proctype = (proctype)theEObject;
				T result = caseproctype(proctype);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.INIT: {
				init init = (init)theEObject;
				T result = caseinit(init);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.NEVER: {
				never never = (never)theEObject;
				T result = casenever(never);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.TRACE: {
				trace trace = (trace)theEObject;
				T result = casetrace(trace);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.UTYPE: {
				utype utype = (utype)theEObject;
				T result = caseutype(utype);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.MTYPE: {
				mtype mtype = (mtype)theEObject;
				T result = casemtype(mtype);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ONE_DECL: {
				one_decl one_decl = (one_decl)theEObject;
				T result = caseone_decl(one_decl);
				if (result == null) result = casestep(one_decl);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.TYPENAME: {
				typename typename = (typename)theEObject;
				T result = casetypename(typename);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.PRIMITIVETYPE: {
				primitivetype primitivetype = (primitivetype)theEObject;
				T result = caseprimitivetype(primitivetype);
				if (result == null) result = casetypename(primitivetype);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.UNAME: {
				uname uname = (uname)theEObject;
				T result = caseuname(uname);
				if (result == null) result = casetypename(uname);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.STEP: {
				step step = (step)theEObject;
				T result = casestep(step);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.STMNT: {
				stmnt stmnt = (stmnt)theEObject;
				T result = casestmnt(stmnt);
				if (result == null) result = casestep(stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.CHANASSERT: {
				chanassert chanassert = (chanassert)theEObject;
				T result = casechanassert(chanassert);
				if (result == null) result = casestep(chanassert);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.IVAR: {
				ivar ivar = (ivar)theEObject;
				T result = caseivar(ivar);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ANY_EXPR_OR_CH_INIT: {
				any_expr_or_ch_init any_expr_or_ch_init = (any_expr_or_ch_init)theEObject;
				T result = caseany_expr_or_ch_init(any_expr_or_ch_init);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.CH_INIT: {
				ch_init ch_init = (ch_init)theEObject;
				T result = casech_init(ch_init);
				if (result == null) result = caseany_expr_or_ch_init(ch_init);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.VARREF: {
				varref varref = (varref)theEObject;
				T result = casevarref(varref);
				if (result == null) result = caseany_expr(varref);
				if (result == null) result = caseexpr(varref);
				if (result == null) result = caseany_expr_or_ch_init(varref);
				if (result == null) result = casestmnt(varref);
				if (result == null) result = casestep(varref);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.SEND: {
				send send = (send)theEObject;
				T result = casesend(send);
				if (result == null) result = casestmnt(send);
				if (result == null) result = casestep(send);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.RECEIVE: {
				receive receive = (receive)theEObject;
				T result = casereceive(receive);
				if (result == null) result = casestmnt(receive);
				if (result == null) result = casestep(receive);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.POLL: {
				poll poll = (poll)theEObject;
				T result = casepoll(poll);
				if (result == null) result = caseany_expr(poll);
				if (result == null) result = caseexpr(poll);
				if (result == null) result = caseany_expr_or_ch_init(poll);
				if (result == null) result = casestmnt(poll);
				if (result == null) result = casestep(poll);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.RECV_ARG: {
				recv_arg recv_arg = (recv_arg)theEObject;
				T result = caserecv_arg(recv_arg);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.RECV_ARG_VARREF: {
				recv_arg_varref recv_arg_varref = (recv_arg_varref)theEObject;
				T result = caserecv_arg_varref(recv_arg_varref);
				if (result == null) result = caserecv_arg(recv_arg_varref);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.RECV_ARG_EVAL_EXPR: {
				recv_arg_eval_expr recv_arg_eval_expr = (recv_arg_eval_expr)theEObject;
				T result = caserecv_arg_eval_expr(recv_arg_eval_expr);
				if (result == null) result = caserecv_arg(recv_arg_eval_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ASSIGN: {
				assign assign = (assign)theEObject;
				T result = caseassign(assign);
				if (result == null) result = casestmnt(assign);
				if (result == null) result = casestep(assign);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ASSIGN_STD: {
				assign_std assign_std = (assign_std)theEObject;
				T result = caseassign_std(assign_std);
				if (result == null) result = caseassign(assign_std);
				if (result == null) result = casestmnt(assign_std);
				if (result == null) result = casestep(assign_std);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ASSIGN_INC: {
				assign_inc assign_inc = (assign_inc)theEObject;
				T result = caseassign_inc(assign_inc);
				if (result == null) result = caseassign(assign_inc);
				if (result == null) result = casestmnt(assign_inc);
				if (result == null) result = casestep(assign_inc);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ASSIGN_DEC: {
				assign_dec assign_dec = (assign_dec)theEObject;
				T result = caseassign_dec(assign_dec);
				if (result == null) result = caseassign(assign_dec);
				if (result == null) result = casestmnt(assign_dec);
				if (result == null) result = casestep(assign_dec);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.IF_STMNT: {
				if_stmnt if_stmnt = (if_stmnt)theEObject;
				T result = caseif_stmnt(if_stmnt);
				if (result == null) result = casestmnt(if_stmnt);
				if (result == null) result = casestep(if_stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.DO_STMNT: {
				do_stmnt do_stmnt = (do_stmnt)theEObject;
				T result = casedo_stmnt(do_stmnt);
				if (result == null) result = casestmnt(do_stmnt);
				if (result == null) result = casestep(do_stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ATOMIC_STMNT: {
				atomic_stmnt atomic_stmnt = (atomic_stmnt)theEObject;
				T result = caseatomic_stmnt(atomic_stmnt);
				if (result == null) result = casestmnt(atomic_stmnt);
				if (result == null) result = casestep(atomic_stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.DSTEP_STMNT: {
				d_step_stmnt d_step_stmnt = (d_step_stmnt)theEObject;
				T result = cased_step_stmnt(d_step_stmnt);
				if (result == null) result = casestmnt(d_step_stmnt);
				if (result == null) result = casestep(d_step_stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.BLOCK_STMNT: {
				block_stmnt block_stmnt = (block_stmnt)theEObject;
				T result = caseblock_stmnt(block_stmnt);
				if (result == null) result = casestmnt(block_stmnt);
				if (result == null) result = casestep(block_stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ELSE_STMNT: {
				else_stmnt else_stmnt = (else_stmnt)theEObject;
				T result = caseelse_stmnt(else_stmnt);
				if (result == null) result = casestmnt(else_stmnt);
				if (result == null) result = casestep(else_stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.BREAK_STMNT: {
				break_stmnt break_stmnt = (break_stmnt)theEObject;
				T result = casebreak_stmnt(break_stmnt);
				if (result == null) result = casestmnt(break_stmnt);
				if (result == null) result = casestep(break_stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.GOTO_STMNT: {
				goto_stmnt goto_stmnt = (goto_stmnt)theEObject;
				T result = casegoto_stmnt(goto_stmnt);
				if (result == null) result = casestmnt(goto_stmnt);
				if (result == null) result = casestep(goto_stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.PRINT_STMNT: {
				print_stmnt print_stmnt = (print_stmnt)theEObject;
				T result = caseprint_stmnt(print_stmnt);
				if (result == null) result = casestmnt(print_stmnt);
				if (result == null) result = casestep(print_stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ASSERT_STMNT: {
				assert_stmnt assert_stmnt = (assert_stmnt)theEObject;
				T result = caseassert_stmnt(assert_stmnt);
				if (result == null) result = casestmnt(assert_stmnt);
				if (result == null) result = casestep(assert_stmnt);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.CCODE: {
				c_code c_code = (c_code)theEObject;
				T result = casec_code(c_code);
				if (result == null) result = casestmnt(c_code);
				if (result == null) result = casestep(c_code);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.CEXPR: {
				c_expr c_expr = (c_expr)theEObject;
				T result = casec_expr(c_expr);
				if (result == null) result = casestmnt(c_expr);
				if (result == null) result = casestep(c_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.CDECL: {
				c_decl c_decl = (c_decl)theEObject;
				T result = casec_decl(c_decl);
				if (result == null) result = casestmnt(c_decl);
				if (result == null) result = casestep(c_decl);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.CTRACK: {
				c_track c_track = (c_track)theEObject;
				T result = casec_track(c_track);
				if (result == null) result = casestmnt(c_track);
				if (result == null) result = casestep(c_track);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.CSTATE: {
				c_state c_state = (c_state)theEObject;
				T result = casec_state(c_state);
				if (result == null) result = casestmnt(c_state);
				if (result == null) result = casestep(c_state);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.OPTIONS: {
				options options = (options)theEObject;
				T result = caseoptions(options);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ANY_EXPR: {
				any_expr any_expr = (any_expr)theEObject;
				T result = caseany_expr(any_expr);
				if (result == null) result = caseexpr(any_expr);
				if (result == null) result = caseany_expr_or_ch_init(any_expr);
				if (result == null) result = casestmnt(any_expr);
				if (result == null) result = casestep(any_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.BIN_EXPR: {
				bin_expr bin_expr = (bin_expr)theEObject;
				T result = casebin_expr(bin_expr);
				if (result == null) result = caseany_expr(bin_expr);
				if (result == null) result = caseexpr(bin_expr);
				if (result == null) result = caseany_expr_or_ch_init(bin_expr);
				if (result == null) result = casestmnt(bin_expr);
				if (result == null) result = casestep(bin_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.UN_EXPR: {
				un_expr un_expr = (un_expr)theEObject;
				T result = caseun_expr(un_expr);
				if (result == null) result = caseany_expr(un_expr);
				if (result == null) result = caseexpr(un_expr);
				if (result == null) result = caseany_expr_or_ch_init(un_expr);
				if (result == null) result = casestmnt(un_expr);
				if (result == null) result = casestep(un_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.COND_EXPR: {
				cond_expr cond_expr = (cond_expr)theEObject;
				T result = casecond_expr(cond_expr);
				if (result == null) result = caseany_expr(cond_expr);
				if (result == null) result = caseexpr(cond_expr);
				if (result == null) result = caseany_expr_or_ch_init(cond_expr);
				if (result == null) result = casestmnt(cond_expr);
				if (result == null) result = casestep(cond_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.LEN_EXPR: {
				len_expr len_expr = (len_expr)theEObject;
				T result = caselen_expr(len_expr);
				if (result == null) result = caseany_expr(len_expr);
				if (result == null) result = caseexpr(len_expr);
				if (result == null) result = caseany_expr_or_ch_init(len_expr);
				if (result == null) result = casestmnt(len_expr);
				if (result == null) result = casestep(len_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.TIMEOUT_EXPR: {
				timeout_expr timeout_expr = (timeout_expr)theEObject;
				T result = casetimeout_expr(timeout_expr);
				if (result == null) result = caseany_expr(timeout_expr);
				if (result == null) result = caseexpr(timeout_expr);
				if (result == null) result = caseany_expr_or_ch_init(timeout_expr);
				if (result == null) result = casestmnt(timeout_expr);
				if (result == null) result = casestep(timeout_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.NP_EXPR: {
				np__expr np__expr = (np__expr)theEObject;
				T result = casenp__expr(np__expr);
				if (result == null) result = caseany_expr(np__expr);
				if (result == null) result = caseexpr(np__expr);
				if (result == null) result = caseany_expr_or_ch_init(np__expr);
				if (result == null) result = casestmnt(np__expr);
				if (result == null) result = casestep(np__expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ENABLED_EXPR: {
				enabled_expr enabled_expr = (enabled_expr)theEObject;
				T result = caseenabled_expr(enabled_expr);
				if (result == null) result = caseany_expr(enabled_expr);
				if (result == null) result = caseexpr(enabled_expr);
				if (result == null) result = caseany_expr_or_ch_init(enabled_expr);
				if (result == null) result = casestmnt(enabled_expr);
				if (result == null) result = casestep(enabled_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.PC_VALUE_EXPR: {
				pc_value_expr pc_value_expr = (pc_value_expr)theEObject;
				T result = casepc_value_expr(pc_value_expr);
				if (result == null) result = caseany_expr(pc_value_expr);
				if (result == null) result = caseexpr(pc_value_expr);
				if (result == null) result = caseany_expr_or_ch_init(pc_value_expr);
				if (result == null) result = casestmnt(pc_value_expr);
				if (result == null) result = casestep(pc_value_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.NAME_EXPR: {
				name_expr name_expr = (name_expr)theEObject;
				T result = casename_expr(name_expr);
				if (result == null) result = caseany_expr(name_expr);
				if (result == null) result = caseexpr(name_expr);
				if (result == null) result = caseany_expr_or_ch_init(name_expr);
				if (result == null) result = casestmnt(name_expr);
				if (result == null) result = casestep(name_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.RUN_EXPR: {
				run_expr run_expr = (run_expr)theEObject;
				T result = caserun_expr(run_expr);
				if (result == null) result = caseany_expr(run_expr);
				if (result == null) result = caseexpr(run_expr);
				if (result == null) result = caseany_expr_or_ch_init(run_expr);
				if (result == null) result = casestmnt(run_expr);
				if (result == null) result = casestep(run_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.EXPR: {
				expr expr = (expr)theEObject;
				T result = caseexpr(expr);
				if (result == null) result = casestmnt(expr);
				if (result == null) result = casestep(expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ANDOR_EXPR: {
				andor_expr andor_expr = (andor_expr)theEObject;
				T result = caseandor_expr(andor_expr);
				if (result == null) result = caseexpr(andor_expr);
				if (result == null) result = casestmnt(andor_expr);
				if (result == null) result = casestep(andor_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.CHANPOLL_EXPR: {
				chanpoll_expr chanpoll_expr = (chanpoll_expr)theEObject;
				T result = casechanpoll_expr(chanpoll_expr);
				if (result == null) result = caseexpr(chanpoll_expr);
				if (result == null) result = casestmnt(chanpoll_expr);
				if (result == null) result = casestep(chanpoll_expr);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.CONSTANT: {
				constant constant = (constant)theEObject;
				T result = caseconstant(constant);
				if (result == null) result = caseany_expr(constant);
				if (result == null) result = caseexpr(constant);
				if (result == null) result = caseany_expr_or_ch_init(constant);
				if (result == null) result = casestmnt(constant);
				if (result == null) result = casestep(constant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.NUM_CONST: {
				num_const num_const = (num_const)theEObject;
				T result = casenum_const(num_const);
				if (result == null) result = caseconstant(num_const);
				if (result == null) result = caserecv_arg(num_const);
				if (result == null) result = caseany_expr(num_const);
				if (result == null) result = caseexpr(num_const);
				if (result == null) result = caseany_expr_or_ch_init(num_const);
				if (result == null) result = casestmnt(num_const);
				if (result == null) result = casestep(num_const);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.ENUM_CONST: {
				enum_const enum_const = (enum_const)theEObject;
				T result = caseenum_const(enum_const);
				if (result == null) result = caseconstant(enum_const);
				if (result == null) result = caseany_expr(enum_const);
				if (result == null) result = caseexpr(enum_const);
				if (result == null) result = caseany_expr_or_ch_init(enum_const);
				if (result == null) result = casestmnt(enum_const);
				if (result == null) result = casestep(enum_const);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case PromelaPackage.MTYPE_CONST: {
				mtype_const mtype_const = (mtype_const)theEObject;
				T result = casemtype_const(mtype_const);
				if (result == null) result = caseconstant(mtype_const);
				if (result == null) result = caserecv_arg(mtype_const);
				if (result == null) result = caseany_expr(mtype_const);
				if (result == null) result = caseexpr(mtype_const);
				if (result == null) result = caseany_expr_or_ch_init(mtype_const);
				if (result == null) result = casestmnt(mtype_const);
				if (result == null) result = casestep(mtype_const);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			default: return defaultCase(theEObject);
		}
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>spec</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>spec</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casespec(spec object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>proctype</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>proctype</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseproctype(proctype object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>init</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>init</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseinit(init object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>never</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>never</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casenever(never object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>trace</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>trace</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casetrace(trace object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>utype</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>utype</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseutype(utype object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>mtype</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>mtype</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casemtype(mtype object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>one decl</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>one decl</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseone_decl(one_decl object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>typename</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>typename</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casetypename(typename object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>primitivetype</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>primitivetype</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseprimitivetype(primitivetype object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>uname</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>uname</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseuname(uname object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>step</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>step</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casestep(step object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casestmnt(stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>chanassert</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>chanassert</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casechanassert(chanassert object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>ivar</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>ivar</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseivar(ivar object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>any expr or ch init</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>any expr or ch init</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseany_expr_or_ch_init(any_expr_or_ch_init object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>ch init</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>ch init</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casech_init(ch_init object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>varref</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>varref</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casevarref(varref object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>send</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>send</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casesend(send object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>receive</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>receive</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casereceive(receive object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>poll</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>poll</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casepoll(poll object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>recv arg</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>recv arg</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caserecv_arg(recv_arg object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>recv arg varref</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>recv arg varref</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caserecv_arg_varref(recv_arg_varref object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>recv arg eval expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>recv arg eval expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caserecv_arg_eval_expr(recv_arg_eval_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>assign</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>assign</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseassign(assign object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>assign std</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>assign std</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseassign_std(assign_std object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>assign inc</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>assign inc</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseassign_inc(assign_inc object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>assign dec</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>assign dec</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseassign_dec(assign_dec object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>if stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>if stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseif_stmnt(if_stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>do stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>do stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casedo_stmnt(do_stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>atomic stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>atomic stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseatomic_stmnt(atomic_stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>dstep stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>dstep stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T cased_step_stmnt(d_step_stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>block stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>block stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseblock_stmnt(block_stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>else stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>else stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseelse_stmnt(else_stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>break stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>break stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casebreak_stmnt(break_stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>goto stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>goto stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casegoto_stmnt(goto_stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>print stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>print stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseprint_stmnt(print_stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>assert stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>assert stmnt</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseassert_stmnt(assert_stmnt object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>ccode</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>ccode</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casec_code(c_code object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>cexpr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>cexpr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casec_expr(c_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>cdecl</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>cdecl</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casec_decl(c_decl object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>ctrack</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>ctrack</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casec_track(c_track object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>cstate</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>cstate</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casec_state(c_state object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>options</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>options</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseoptions(options object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>any expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>any expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseany_expr(any_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>bin expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>bin expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casebin_expr(bin_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>un expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>un expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseun_expr(un_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>cond expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>cond expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casecond_expr(cond_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>len expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>len expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caselen_expr(len_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>timeout expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>timeout expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casetimeout_expr(timeout_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>np expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>np expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casenp__expr(np__expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>enabled expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>enabled expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseenabled_expr(enabled_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>pc value expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>pc value expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casepc_value_expr(pc_value_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>name expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>name expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casename_expr(name_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>run expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>run expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caserun_expr(run_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseexpr(expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>andor expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>andor expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseandor_expr(andor_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>chanpoll expr</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>chanpoll expr</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casechanpoll_expr(chanpoll_expr object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>constant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>constant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseconstant(constant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>num const</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>num const</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casenum_const(num_const object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>enum const</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>enum const</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseenum_const(enum_const object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>mtype const</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>mtype const</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casemtype_const(mtype_const object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>EObject</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch, but this is the last case anyway.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>EObject</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject)
	 * @generated
	 */
	@Override
	public T defaultCase(EObject object) {
		return null;
	}

} //PromelaSwitch
