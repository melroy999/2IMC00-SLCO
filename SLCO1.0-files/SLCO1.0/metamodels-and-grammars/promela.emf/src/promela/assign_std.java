/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>assign std</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.assign_std#getVarref <em>Varref</em>}</li>
 *   <li>{@link promela.assign_std#getAny_expr <em>Any expr</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getassign_std()
 * @model
 * @generated
 */
public interface assign_std extends assign {
	/**
	 * Returns the value of the '<em><b>Varref</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Varref</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Varref</em>' containment reference.
	 * @see #setVarref(varref)
	 * @see promela.PromelaPackage#getassign_std_Varref()
	 * @model containment="true" required="true"
	 * @generated
	 */
	varref getVarref();

	/**
	 * Sets the value of the '{@link promela.assign_std#getVarref <em>Varref</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Varref</em>' containment reference.
	 * @see #getVarref()
	 * @generated
	 */
	void setVarref(varref value);

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
	 * @see promela.PromelaPackage#getassign_std_Any_expr()
	 * @model containment="true" required="true"
	 * @generated
	 */
	any_expr getAny_expr();

	/**
	 * Sets the value of the '{@link promela.assign_std#getAny_expr <em>Any expr</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Any expr</em>' containment reference.
	 * @see #getAny_expr()
	 * @generated
	 */
	void setAny_expr(any_expr value);

} // assign_std
