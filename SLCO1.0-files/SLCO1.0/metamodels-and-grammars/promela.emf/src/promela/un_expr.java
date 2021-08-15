/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>un expr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.un_expr#getOperator <em>Operator</em>}</li>
 *   <li>{@link promela.un_expr#getOperand <em>Operand</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getun_expr()
 * @model
 * @generated
 */
public interface un_expr extends any_expr {
	/**
	 * Returns the value of the '<em><b>Operator</b></em>' attribute.
	 * The literals are from the enumeration {@link promela.unarop}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operator</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operator</em>' attribute.
	 * @see promela.unarop
	 * @see #setOperator(unarop)
	 * @see promela.PromelaPackage#getun_expr_Operator()
	 * @model required="true"
	 * @generated
	 */
	unarop getOperator();

	/**
	 * Sets the value of the '{@link promela.un_expr#getOperator <em>Operator</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operator</em>' attribute.
	 * @see promela.unarop
	 * @see #getOperator()
	 * @generated
	 */
	void setOperator(unarop value);

	/**
	 * Returns the value of the '<em><b>Operand</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operand</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operand</em>' containment reference.
	 * @see #setOperand(any_expr)
	 * @see promela.PromelaPackage#getun_expr_Operand()
	 * @model containment="true" required="true"
	 * @generated
	 */
	any_expr getOperand();

	/**
	 * Sets the value of the '{@link promela.un_expr#getOperand <em>Operand</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operand</em>' containment reference.
	 * @see #getOperand()
	 * @generated
	 */
	void setOperand(any_expr value);

} // un_expr
