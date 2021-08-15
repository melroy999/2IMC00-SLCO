/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @see promela.PromelaPackage
 * @generated
 */
public interface PromelaFactory extends EFactory {
	/**
	 * The singleton instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	PromelaFactory eINSTANCE = promela.impl.PromelaFactoryImpl.init();

	/**
	 * Returns a new object of class '<em>spec</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>spec</em>'.
	 * @generated
	 */
	spec createspec();

	/**
	 * Returns a new object of class '<em>proctype</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>proctype</em>'.
	 * @generated
	 */
	proctype createproctype();

	/**
	 * Returns a new object of class '<em>init</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>init</em>'.
	 * @generated
	 */
	init createinit();

	/**
	 * Returns a new object of class '<em>never</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>never</em>'.
	 * @generated
	 */
	never createnever();

	/**
	 * Returns a new object of class '<em>trace</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>trace</em>'.
	 * @generated
	 */
	trace createtrace();

	/**
	 * Returns a new object of class '<em>utype</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>utype</em>'.
	 * @generated
	 */
	utype createutype();

	/**
	 * Returns a new object of class '<em>mtype</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>mtype</em>'.
	 * @generated
	 */
	mtype createmtype();

	/**
	 * Returns a new object of class '<em>one decl</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>one decl</em>'.
	 * @generated
	 */
	one_decl createone_decl();

	/**
	 * Returns a new object of class '<em>primitivetype</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>primitivetype</em>'.
	 * @generated
	 */
	primitivetype createprimitivetype();

	/**
	 * Returns a new object of class '<em>uname</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>uname</em>'.
	 * @generated
	 */
	uname createuname();

	/**
	 * Returns a new object of class '<em>chanassert</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>chanassert</em>'.
	 * @generated
	 */
	chanassert createchanassert();

	/**
	 * Returns a new object of class '<em>ivar</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>ivar</em>'.
	 * @generated
	 */
	ivar createivar();

	/**
	 * Returns a new object of class '<em>ch init</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>ch init</em>'.
	 * @generated
	 */
	ch_init createch_init();

	/**
	 * Returns a new object of class '<em>varref</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>varref</em>'.
	 * @generated
	 */
	varref createvarref();

	/**
	 * Returns a new object of class '<em>send</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>send</em>'.
	 * @generated
	 */
	send createsend();

	/**
	 * Returns a new object of class '<em>receive</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>receive</em>'.
	 * @generated
	 */
	receive createreceive();

	/**
	 * Returns a new object of class '<em>recv arg varref</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>recv arg varref</em>'.
	 * @generated
	 */
	recv_arg_varref createrecv_arg_varref();

	/**
	 * Returns a new object of class '<em>recv arg eval expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>recv arg eval expr</em>'.
	 * @generated
	 */
	recv_arg_eval_expr createrecv_arg_eval_expr();

	/**
	 * Returns a new object of class '<em>assign std</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>assign std</em>'.
	 * @generated
	 */
	assign_std createassign_std();

	/**
	 * Returns a new object of class '<em>assign inc</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>assign inc</em>'.
	 * @generated
	 */
	assign_inc createassign_inc();

	/**
	 * Returns a new object of class '<em>assign dec</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>assign dec</em>'.
	 * @generated
	 */
	assign_dec createassign_dec();

	/**
	 * Returns a new object of class '<em>if stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>if stmnt</em>'.
	 * @generated
	 */
	if_stmnt createif_stmnt();

	/**
	 * Returns a new object of class '<em>do stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>do stmnt</em>'.
	 * @generated
	 */
	do_stmnt createdo_stmnt();

	/**
	 * Returns a new object of class '<em>atomic stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>atomic stmnt</em>'.
	 * @generated
	 */
	atomic_stmnt createatomic_stmnt();

	/**
	 * Returns a new object of class '<em>dstep stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>dstep stmnt</em>'.
	 * @generated
	 */
	d_step_stmnt created_step_stmnt();

	/**
	 * Returns a new object of class '<em>block stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>block stmnt</em>'.
	 * @generated
	 */
	block_stmnt createblock_stmnt();

	/**
	 * Returns a new object of class '<em>else stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>else stmnt</em>'.
	 * @generated
	 */
	else_stmnt createelse_stmnt();

	/**
	 * Returns a new object of class '<em>break stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>break stmnt</em>'.
	 * @generated
	 */
	break_stmnt createbreak_stmnt();

	/**
	 * Returns a new object of class '<em>goto stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>goto stmnt</em>'.
	 * @generated
	 */
	goto_stmnt creategoto_stmnt();

	/**
	 * Returns a new object of class '<em>print stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>print stmnt</em>'.
	 * @generated
	 */
	print_stmnt createprint_stmnt();

	/**
	 * Returns a new object of class '<em>assert stmnt</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>assert stmnt</em>'.
	 * @generated
	 */
	assert_stmnt createassert_stmnt();

	/**
	 * Returns a new object of class '<em>ccode</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>ccode</em>'.
	 * @generated
	 */
	c_code createc_code();

	/**
	 * Returns a new object of class '<em>cexpr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>cexpr</em>'.
	 * @generated
	 */
	c_expr createc_expr();

	/**
	 * Returns a new object of class '<em>cdecl</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>cdecl</em>'.
	 * @generated
	 */
	c_decl createc_decl();

	/**
	 * Returns a new object of class '<em>ctrack</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>ctrack</em>'.
	 * @generated
	 */
	c_track createc_track();

	/**
	 * Returns a new object of class '<em>cstate</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>cstate</em>'.
	 * @generated
	 */
	c_state createc_state();

	/**
	 * Returns a new object of class '<em>options</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>options</em>'.
	 * @generated
	 */
	options createoptions();

	/**
	 * Returns a new object of class '<em>bin expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>bin expr</em>'.
	 * @generated
	 */
	bin_expr createbin_expr();

	/**
	 * Returns a new object of class '<em>un expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>un expr</em>'.
	 * @generated
	 */
	un_expr createun_expr();

	/**
	 * Returns a new object of class '<em>cond expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>cond expr</em>'.
	 * @generated
	 */
	cond_expr createcond_expr();

	/**
	 * Returns a new object of class '<em>len expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>len expr</em>'.
	 * @generated
	 */
	len_expr createlen_expr();

	/**
	 * Returns a new object of class '<em>timeout expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>timeout expr</em>'.
	 * @generated
	 */
	timeout_expr createtimeout_expr();

	/**
	 * Returns a new object of class '<em>np expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>np expr</em>'.
	 * @generated
	 */
	np__expr createnp__expr();

	/**
	 * Returns a new object of class '<em>enabled expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>enabled expr</em>'.
	 * @generated
	 */
	enabled_expr createenabled_expr();

	/**
	 * Returns a new object of class '<em>pc value expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>pc value expr</em>'.
	 * @generated
	 */
	pc_value_expr createpc_value_expr();

	/**
	 * Returns a new object of class '<em>name expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>name expr</em>'.
	 * @generated
	 */
	name_expr createname_expr();

	/**
	 * Returns a new object of class '<em>run expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>run expr</em>'.
	 * @generated
	 */
	run_expr createrun_expr();

	/**
	 * Returns a new object of class '<em>andor expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>andor expr</em>'.
	 * @generated
	 */
	andor_expr createandor_expr();

	/**
	 * Returns a new object of class '<em>chanpoll expr</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>chanpoll expr</em>'.
	 * @generated
	 */
	chanpoll_expr createchanpoll_expr();

	/**
	 * Returns a new object of class '<em>num const</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>num const</em>'.
	 * @generated
	 */
	num_const createnum_const();

	/**
	 * Returns a new object of class '<em>enum const</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>enum const</em>'.
	 * @generated
	 */
	enum_const createenum_const();

	/**
	 * Returns a new object of class '<em>mtype const</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>mtype const</em>'.
	 * @generated
	 */
	mtype_const createmtype_const();

	/**
	 * Returns the package supported by this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package supported by this factory.
	 * @generated
	 */
	PromelaPackage getPromelaPackage();

} //PromelaFactory
