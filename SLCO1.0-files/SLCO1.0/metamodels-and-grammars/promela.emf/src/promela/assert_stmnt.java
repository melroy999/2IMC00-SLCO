/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>assert stmnt</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.assert_stmnt#getExpr <em>Expr</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getassert_stmnt()
 * @model
 * @generated
 */
public interface assert_stmnt extends stmnt {
	/**
	 * Returns the value of the '<em><b>Expr</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Expr</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Expr</em>' containment reference.
	 * @see #setExpr(expr)
	 * @see promela.PromelaPackage#getassert_stmnt_Expr()
	 * @model containment="true" required="true"
	 * @generated
	 */
	expr getExpr();

	/**
	 * Sets the value of the '{@link promela.assert_stmnt#getExpr <em>Expr</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Expr</em>' containment reference.
	 * @see #getExpr()
	 * @generated
	 */
	void setExpr(expr value);

} // assert_stmnt
