/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>pc value expr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.pc_value_expr#getAny_expr <em>Any expr</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getpc_value_expr()
 * @model
 * @generated
 */
public interface pc_value_expr extends any_expr {
	/**
	 * Returns the value of the '<em><b>Any expr</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Any expr</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Any expr</em>' containment reference.
	 * @see #setAny_expr(any_expr)
	 * @see promela.PromelaPackage#getpc_value_expr_Any_expr()
	 * @model containment="true" required="true"
	 * @generated
	 */
	any_expr getAny_expr();

	/**
	 * Sets the value of the '{@link promela.pc_value_expr#getAny_expr <em>Any expr</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Any expr</em>' containment reference.
	 * @see #getAny_expr()
	 * @generated
	 */
	void setAny_expr(any_expr value);

} // pc_value_expr
