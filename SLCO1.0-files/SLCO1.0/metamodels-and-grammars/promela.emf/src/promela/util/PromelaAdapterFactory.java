/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.util;

import org.eclipse.emf.common.notify.Adapter;
import org.eclipse.emf.common.notify.Notifier;

import org.eclipse.emf.common.notify.impl.AdapterFactoryImpl;

import org.eclipse.emf.ecore.EObject;

import promela.*;

/**
 * <!-- begin-user-doc -->
 * The <b>Adapter Factory</b> for the model.
 * It provides an adapter <code>createXXX</code> method for each class of the model.
 * <!-- end-user-doc -->
 * @see promela.PromelaPackage
 * @generated
 */
public class PromelaAdapterFactory extends AdapterFactoryImpl {
	/**
	 * The cached model package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static PromelaPackage modelPackage;

	/**
	 * Creates an instance of the adapter factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PromelaAdapterFactory() {
		if (modelPackage == null) {
			modelPackage = PromelaPackage.eINSTANCE;
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
	protected PromelaSwitch<Adapter> modelSwitch =
		new PromelaSwitch<Adapter>() {
			@Override
			public Adapter casespec(spec object) {
				return createspecAdapter();
			}
			@Override
			public Adapter caseproctype(proctype object) {
				return createproctypeAdapter();
			}
			@Override
			public Adapter caseinit(init object) {
				return createinitAdapter();
			}
			@Override
			public Adapter casenever(never object) {
				return createneverAdapter();
			}
			@Override
			public Adapter casetrace(trace object) {
				return createtraceAdapter();
			}
			@Override
			public Adapter caseutype(utype object) {
				return createutypeAdapter();
			}
			@Override
			public Adapter casemtype(mtype object) {
				return createmtypeAdapter();
			}
			@Override
			public Adapter caseone_decl(one_decl object) {
				return createone_declAdapter();
			}
			@Override
			public Adapter casetypename(typename object) {
				return createtypenameAdapter();
			}
			@Override
			public Adapter caseprimitivetype(primitivetype object) {
				return createprimitivetypeAdapter();
			}
			@Override
			public Adapter caseuname(uname object) {
				return createunameAdapter();
			}
			@Override
			public Adapter casestep(step object) {
				return createstepAdapter();
			}
			@Override
			public Adapter casestmnt(stmnt object) {
				return createstmntAdapter();
			}
			@Override
			public Adapter casechanassert(chanassert object) {
				return createchanassertAdapter();
			}
			@Override
			public Adapter caseivar(ivar object) {
				return createivarAdapter();
			}
			@Override
			public Adapter caseany_expr_or_ch_init(any_expr_or_ch_init object) {
				return createany_expr_or_ch_initAdapter();
			}
			@Override
			public Adapter casech_init(ch_init object) {
				return createch_initAdapter();
			}
			@Override
			public Adapter casevarref(varref object) {
				return createvarrefAdapter();
			}
			@Override
			public Adapter casesend(send object) {
				return createsendAdapter();
			}
			@Override
			public Adapter casereceive(receive object) {
				return createreceiveAdapter();
			}
			@Override
			public Adapter casepoll(poll object) {
				return createpollAdapter();
			}
			@Override
			public Adapter caserecv_arg(recv_arg object) {
				return createrecv_argAdapter();
			}
			@Override
			public Adapter caserecv_arg_varref(recv_arg_varref object) {
				return createrecv_arg_varrefAdapter();
			}
			@Override
			public Adapter caserecv_arg_eval_expr(recv_arg_eval_expr object) {
				return createrecv_arg_eval_exprAdapter();
			}
			@Override
			public Adapter caseassign(assign object) {
				return createassignAdapter();
			}
			@Override
			public Adapter caseassign_std(assign_std object) {
				return createassign_stdAdapter();
			}
			@Override
			public Adapter caseassign_inc(assign_inc object) {
				return createassign_incAdapter();
			}
			@Override
			public Adapter caseassign_dec(assign_dec object) {
				return createassign_decAdapter();
			}
			@Override
			public Adapter caseif_stmnt(if_stmnt object) {
				return createif_stmntAdapter();
			}
			@Override
			public Adapter casedo_stmnt(do_stmnt object) {
				return createdo_stmntAdapter();
			}
			@Override
			public Adapter caseatomic_stmnt(atomic_stmnt object) {
				return createatomic_stmntAdapter();
			}
			@Override
			public Adapter cased_step_stmnt(d_step_stmnt object) {
				return created_step_stmntAdapter();
			}
			@Override
			public Adapter caseblock_stmnt(block_stmnt object) {
				return createblock_stmntAdapter();
			}
			@Override
			public Adapter caseelse_stmnt(else_stmnt object) {
				return createelse_stmntAdapter();
			}
			@Override
			public Adapter casebreak_stmnt(break_stmnt object) {
				return createbreak_stmntAdapter();
			}
			@Override
			public Adapter casegoto_stmnt(goto_stmnt object) {
				return creategoto_stmntAdapter();
			}
			@Override
			public Adapter caseprint_stmnt(print_stmnt object) {
				return createprint_stmntAdapter();
			}
			@Override
			public Adapter caseassert_stmnt(assert_stmnt object) {
				return createassert_stmntAdapter();
			}
			@Override
			public Adapter casec_code(c_code object) {
				return createc_codeAdapter();
			}
			@Override
			public Adapter casec_expr(c_expr object) {
				return createc_exprAdapter();
			}
			@Override
			public Adapter casec_decl(c_decl object) {
				return createc_declAdapter();
			}
			@Override
			public Adapter casec_track(c_track object) {
				return createc_trackAdapter();
			}
			@Override
			public Adapter casec_state(c_state object) {
				return createc_stateAdapter();
			}
			@Override
			public Adapter caseoptions(options object) {
				return createoptionsAdapter();
			}
			@Override
			public Adapter caseany_expr(any_expr object) {
				return createany_exprAdapter();
			}
			@Override
			public Adapter casebin_expr(bin_expr object) {
				return createbin_exprAdapter();
			}
			@Override
			public Adapter caseun_expr(un_expr object) {
				return createun_exprAdapter();
			}
			@Override
			public Adapter casecond_expr(cond_expr object) {
				return createcond_exprAdapter();
			}
			@Override
			public Adapter caselen_expr(len_expr object) {
				return createlen_exprAdapter();
			}
			@Override
			public Adapter casetimeout_expr(timeout_expr object) {
				return createtimeout_exprAdapter();
			}
			@Override
			public Adapter casenp__expr(np__expr object) {
				return createnp__exprAdapter();
			}
			@Override
			public Adapter caseenabled_expr(enabled_expr object) {
				return createenabled_exprAdapter();
			}
			@Override
			public Adapter casepc_value_expr(pc_value_expr object) {
				return createpc_value_exprAdapter();
			}
			@Override
			public Adapter casename_expr(name_expr object) {
				return createname_exprAdapter();
			}
			@Override
			public Adapter caserun_expr(run_expr object) {
				return createrun_exprAdapter();
			}
			@Override
			public Adapter caseexpr(expr object) {
				return createexprAdapter();
			}
			@Override
			public Adapter caseandor_expr(andor_expr object) {
				return createandor_exprAdapter();
			}
			@Override
			public Adapter casechanpoll_expr(chanpoll_expr object) {
				return createchanpoll_exprAdapter();
			}
			@Override
			public Adapter caseconstant(constant object) {
				return createconstantAdapter();
			}
			@Override
			public Adapter casenum_const(num_const object) {
				return createnum_constAdapter();
			}
			@Override
			public Adapter caseenum_const(enum_const object) {
				return createenum_constAdapter();
			}
			@Override
			public Adapter casemtype_const(mtype_const object) {
				return createmtype_constAdapter();
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
	 * Creates a new adapter for an object of class '{@link promela.spec <em>spec</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.spec
	 * @generated
	 */
	public Adapter createspecAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.proctype <em>proctype</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.proctype
	 * @generated
	 */
	public Adapter createproctypeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.init <em>init</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.init
	 * @generated
	 */
	public Adapter createinitAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.never <em>never</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.never
	 * @generated
	 */
	public Adapter createneverAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.trace <em>trace</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.trace
	 * @generated
	 */
	public Adapter createtraceAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.utype <em>utype</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.utype
	 * @generated
	 */
	public Adapter createutypeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.mtype <em>mtype</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.mtype
	 * @generated
	 */
	public Adapter createmtypeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.one_decl <em>one decl</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.one_decl
	 * @generated
	 */
	public Adapter createone_declAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.typename <em>typename</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.typename
	 * @generated
	 */
	public Adapter createtypenameAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.primitivetype <em>primitivetype</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.primitivetype
	 * @generated
	 */
	public Adapter createprimitivetypeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.uname <em>uname</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.uname
	 * @generated
	 */
	public Adapter createunameAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.step <em>step</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.step
	 * @generated
	 */
	public Adapter createstepAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.stmnt <em>stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.stmnt
	 * @generated
	 */
	public Adapter createstmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.chanassert <em>chanassert</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.chanassert
	 * @generated
	 */
	public Adapter createchanassertAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.ivar <em>ivar</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.ivar
	 * @generated
	 */
	public Adapter createivarAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.any_expr_or_ch_init <em>any expr or ch init</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.any_expr_or_ch_init
	 * @generated
	 */
	public Adapter createany_expr_or_ch_initAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.ch_init <em>ch init</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.ch_init
	 * @generated
	 */
	public Adapter createch_initAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.varref <em>varref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.varref
	 * @generated
	 */
	public Adapter createvarrefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.send <em>send</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.send
	 * @generated
	 */
	public Adapter createsendAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.receive <em>receive</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.receive
	 * @generated
	 */
	public Adapter createreceiveAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.poll <em>poll</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.poll
	 * @generated
	 */
	public Adapter createpollAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.recv_arg <em>recv arg</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.recv_arg
	 * @generated
	 */
	public Adapter createrecv_argAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.recv_arg_varref <em>recv arg varref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.recv_arg_varref
	 * @generated
	 */
	public Adapter createrecv_arg_varrefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.recv_arg_eval_expr <em>recv arg eval expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.recv_arg_eval_expr
	 * @generated
	 */
	public Adapter createrecv_arg_eval_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.assign <em>assign</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.assign
	 * @generated
	 */
	public Adapter createassignAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.assign_std <em>assign std</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.assign_std
	 * @generated
	 */
	public Adapter createassign_stdAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.assign_inc <em>assign inc</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.assign_inc
	 * @generated
	 */
	public Adapter createassign_incAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.assign_dec <em>assign dec</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.assign_dec
	 * @generated
	 */
	public Adapter createassign_decAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.if_stmnt <em>if stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.if_stmnt
	 * @generated
	 */
	public Adapter createif_stmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.do_stmnt <em>do stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.do_stmnt
	 * @generated
	 */
	public Adapter createdo_stmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.atomic_stmnt <em>atomic stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.atomic_stmnt
	 * @generated
	 */
	public Adapter createatomic_stmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.d_step_stmnt <em>dstep stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.d_step_stmnt
	 * @generated
	 */
	public Adapter created_step_stmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.block_stmnt <em>block stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.block_stmnt
	 * @generated
	 */
	public Adapter createblock_stmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.else_stmnt <em>else stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.else_stmnt
	 * @generated
	 */
	public Adapter createelse_stmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.break_stmnt <em>break stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.break_stmnt
	 * @generated
	 */
	public Adapter createbreak_stmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.goto_stmnt <em>goto stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.goto_stmnt
	 * @generated
	 */
	public Adapter creategoto_stmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.print_stmnt <em>print stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.print_stmnt
	 * @generated
	 */
	public Adapter createprint_stmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.assert_stmnt <em>assert stmnt</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.assert_stmnt
	 * @generated
	 */
	public Adapter createassert_stmntAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.c_code <em>ccode</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.c_code
	 * @generated
	 */
	public Adapter createc_codeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.c_expr <em>cexpr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.c_expr
	 * @generated
	 */
	public Adapter createc_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.c_decl <em>cdecl</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.c_decl
	 * @generated
	 */
	public Adapter createc_declAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.c_track <em>ctrack</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.c_track
	 * @generated
	 */
	public Adapter createc_trackAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.c_state <em>cstate</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.c_state
	 * @generated
	 */
	public Adapter createc_stateAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.options <em>options</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.options
	 * @generated
	 */
	public Adapter createoptionsAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.any_expr <em>any expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.any_expr
	 * @generated
	 */
	public Adapter createany_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.bin_expr <em>bin expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.bin_expr
	 * @generated
	 */
	public Adapter createbin_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.un_expr <em>un expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.un_expr
	 * @generated
	 */
	public Adapter createun_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.cond_expr <em>cond expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.cond_expr
	 * @generated
	 */
	public Adapter createcond_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.len_expr <em>len expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.len_expr
	 * @generated
	 */
	public Adapter createlen_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.timeout_expr <em>timeout expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.timeout_expr
	 * @generated
	 */
	public Adapter createtimeout_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.np__expr <em>np expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.np__expr
	 * @generated
	 */
	public Adapter createnp__exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.enabled_expr <em>enabled expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.enabled_expr
	 * @generated
	 */
	public Adapter createenabled_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.pc_value_expr <em>pc value expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.pc_value_expr
	 * @generated
	 */
	public Adapter createpc_value_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.name_expr <em>name expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.name_expr
	 * @generated
	 */
	public Adapter createname_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.run_expr <em>run expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.run_expr
	 * @generated
	 */
	public Adapter createrun_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.expr <em>expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.expr
	 * @generated
	 */
	public Adapter createexprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.andor_expr <em>andor expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.andor_expr
	 * @generated
	 */
	public Adapter createandor_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.chanpoll_expr <em>chanpoll expr</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.chanpoll_expr
	 * @generated
	 */
	public Adapter createchanpoll_exprAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.constant <em>constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.constant
	 * @generated
	 */
	public Adapter createconstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.num_const <em>num const</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.num_const
	 * @generated
	 */
	public Adapter createnum_constAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.enum_const <em>enum const</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.enum_const
	 * @generated
	 */
	public Adapter createenum_constAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link promela.mtype_const <em>mtype const</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see promela.mtype_const
	 * @generated
	 */
	public Adapter createmtype_constAdapter() {
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

} //PromelaAdapterFactory
