/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>name expr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.name_expr#getName <em>Name</em>}</li>
 *   <li>{@link promela.name_expr#getAny_expr <em>Any expr</em>}</li>
 *   <li>{@link promela.name_expr#getAt_name <em>At name</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getname_expr()
 * @model
 * @generated
 */
public interface name_expr extends any_expr {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see promela.PromelaPackage#getname_expr_Name()
	 * @model required="true"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link promela.name_expr#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

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
	 * @see promela.PromelaPackage#getname_expr_Any_expr()
	 * @model containment="true"
	 * @generated
	 */
	any_expr getAny_expr();

	/**
	 * Sets the value of the '{@link promela.name_expr#getAny_expr <em>Any expr</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Any expr</em>' containment reference.
	 * @see #getAny_expr()
	 * @generated
	 */
	void setAny_expr(any_expr value);

	/**
	 * Returns the value of the '<em><b>At name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>At name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>At name</em>' attribute.
	 * @see #setAt_name(String)
	 * @see promela.PromelaPackage#getname_expr_At_name()
	 * @model required="true"
	 * @generated
	 */
	String getAt_name();

	/**
	 * Sets the value of the '{@link promela.name_expr#getAt_name <em>At name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>At name</em>' attribute.
	 * @see #getAt_name()
	 * @generated
	 */
	void setAt_name(String value);

} // name_expr
