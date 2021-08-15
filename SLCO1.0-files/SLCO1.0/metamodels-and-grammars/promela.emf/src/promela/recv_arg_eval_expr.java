/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>recv arg eval expr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.recv_arg_eval_expr#getExpr <em>Expr</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getrecv_arg_eval_expr()
 * @model
 * @generated
 */
public interface recv_arg_eval_expr extends recv_arg {
	/**
	 * Returns the value of the '<em><b>Expr</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Expr</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Expr</em>' containment reference.
	 * @see #setExpr(any_expr)
	 * @see promela.PromelaPackage#getrecv_arg_eval_expr_Expr()
	 * @model containment="true" required="true"
	 * @generated
	 */
	any_expr getExpr();

	/**
	 * Sets the value of the '{@link promela.recv_arg_eval_expr#getExpr <em>Expr</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Expr</em>' containment reference.
	 * @see #getExpr()
	 * @generated
	 */
	void setExpr(any_expr value);

} // recv_arg_eval_expr
