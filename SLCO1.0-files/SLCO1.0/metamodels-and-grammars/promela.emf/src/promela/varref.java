/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>varref</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.varref#getVar_name <em>Var name</em>}</li>
 *   <li>{@link promela.varref#getAny_expr <em>Any expr</em>}</li>
 *   <li>{@link promela.varref#getVarref <em>Varref</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getvarref()
 * @model
 * @generated
 */
public interface varref extends any_expr {
	/**
	 * Returns the value of the '<em><b>Var name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Var name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Var name</em>' attribute.
	 * @see #setVar_name(String)
	 * @see promela.PromelaPackage#getvarref_Var_name()
	 * @model required="true"
	 * @generated
	 */
	String getVar_name();

	/**
	 * Sets the value of the '{@link promela.varref#getVar_name <em>Var name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Var name</em>' attribute.
	 * @see #getVar_name()
	 * @generated
	 */
	void setVar_name(String value);

	/**
	 * Returns the value of the '<em><b>Any expr</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Any expr</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Any expr</em>' reference.
	 * @see #setAny_expr(any_expr)
	 * @see promela.PromelaPackage#getvarref_Any_expr()
	 * @model
	 * @generated
	 */
	any_expr getAny_expr();

	/**
	 * Sets the value of the '{@link promela.varref#getAny_expr <em>Any expr</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Any expr</em>' reference.
	 * @see #getAny_expr()
	 * @generated
	 */
	void setAny_expr(any_expr value);

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
	 * @see promela.PromelaPackage#getvarref_Varref()
	 * @model
	 * @generated
	 */
	varref getVarref();

	/**
	 * Sets the value of the '{@link promela.varref#getVarref <em>Varref</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Varref</em>' reference.
	 * @see #getVarref()
	 * @generated
	 */
	void setVarref(varref value);

} // varref
