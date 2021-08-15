/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>andor expr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.andor_expr#getOperator <em>Operator</em>}</li>
 *   <li>{@link promela.andor_expr#getOperand1 <em>Operand1</em>}</li>
 *   <li>{@link promela.andor_expr#getOperand2 <em>Operand2</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getandor_expr()
 * @model
 * @generated
 */
public interface andor_expr extends expr {
	/**
	 * Returns the value of the '<em><b>Operator</b></em>' attribute.
	 * The literals are from the enumeration {@link promela.andor}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operator</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operator</em>' attribute.
	 * @see promela.andor
	 * @see #setOperator(andor)
	 * @see promela.PromelaPackage#getandor_expr_Operator()
	 * @model required="true"
	 * @generated
	 */
	andor getOperator();

	/**
	 * Sets the value of the '{@link promela.andor_expr#getOperator <em>Operator</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operator</em>' attribute.
	 * @see promela.andor
	 * @see #getOperator()
	 * @generated
	 */
	void setOperator(andor value);

	/**
	 * Returns the value of the '<em><b>Operand1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operand1</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operand1</em>' containment reference.
	 * @see #setOperand1(expr)
	 * @see promela.PromelaPackage#getandor_expr_Operand1()
	 * @model containment="true" required="true"
	 * @generated
	 */
	expr getOperand1();

	/**
	 * Sets the value of the '{@link promela.andor_expr#getOperand1 <em>Operand1</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operand1</em>' containment reference.
	 * @see #getOperand1()
	 * @generated
	 */
	void setOperand1(expr value);

	/**
	 * Returns the value of the '<em><b>Operand2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operand2</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operand2</em>' containment reference.
	 * @see #setOperand2(expr)
	 * @see promela.PromelaPackage#getandor_expr_Operand2()
	 * @model containment="true" required="true"
	 * @generated
	 */
	expr getOperand2();

	/**
	 * Sets the value of the '{@link promela.andor_expr#getOperand2 <em>Operand2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operand2</em>' containment reference.
	 * @see #getOperand2()
	 * @generated
	 */
	void setOperand2(expr value);

} // andor_expr
