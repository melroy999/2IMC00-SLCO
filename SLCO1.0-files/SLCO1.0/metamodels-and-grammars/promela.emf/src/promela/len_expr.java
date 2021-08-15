/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>len expr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.len_expr#getVarref <em>Varref</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getlen_expr()
 * @model
 * @generated
 */
public interface len_expr extends any_expr {
	/**
	 * Returns the value of the '<em><b>Varref</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Varref</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Varref</em>' reference.
	 * @see #setVarref(varref)
	 * @see promela.PromelaPackage#getlen_expr_Varref()
	 * @model required="true"
	 * @generated
	 */
	varref getVarref();

	/**
	 * Sets the value of the '{@link promela.len_expr#getVarref <em>Varref</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Varref</em>' reference.
	 * @see #getVarref()
	 * @generated
	 */
	void setVarref(varref value);

} // len_expr
