/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>cond expr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.cond_expr#getExpr1 <em>Expr1</em>}</li>
 *   <li>{@link promela.cond_expr#getExpr2 <em>Expr2</em>}</li>
 *   <li>{@link promela.cond_expr#getExpr3 <em>Expr3</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getcond_expr()
 * @model
 * @generated
 */
public interface cond_expr extends any_expr {
	/**
	 * Returns the value of the '<em><b>Expr1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Expr1</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Expr1</em>' containment reference.
	 * @see #setExpr1(any_expr)
	 * @see promela.PromelaPackage#getcond_expr_Expr1()
	 * @model containment="true" required="true"
	 * @generated
	 */
	any_expr getExpr1();

	/**
	 * Sets the value of the '{@link promela.cond_expr#getExpr1 <em>Expr1</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Expr1</em>' containment reference.
	 * @see #getExpr1()
	 * @generated
	 */
	void setExpr1(any_expr value);

	/**
	 * Returns the value of the '<em><b>Expr2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Expr2</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Expr2</em>' containment reference.
	 * @see #setExpr2(any_expr)
	 * @see promela.PromelaPackage#getcond_expr_Expr2()
	 * @model containment="true" required="true"
	 * @generated
	 */
	any_expr getExpr2();

	/**
	 * Sets the value of the '{@link promela.cond_expr#getExpr2 <em>Expr2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Expr2</em>' containment reference.
	 * @see #getExpr2()
	 * @generated
	 */
	void setExpr2(any_expr value);

	/**
	 * Returns the value of the '<em><b>Expr3</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Expr3</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Expr3</em>' containment reference.
	 * @see #setExpr3(any_expr)
	 * @see promela.PromelaPackage#getcond_expr_Expr3()
	 * @model containment="true" required="true"
	 * @generated
	 */
	any_expr getExpr3();

	/**
	 * Sets the value of the '{@link promela.cond_expr#getExpr3 <em>Expr3</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Expr3</em>' containment reference.
	 * @see #getExpr3()
	 * @generated
	 */
	void setExpr3(any_expr value);

} // cond_expr
