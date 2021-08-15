/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>bin expr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.bin_expr#getOperator <em>Operator</em>}</li>
 *   <li>{@link promela.bin_expr#getOperand1 <em>Operand1</em>}</li>
 *   <li>{@link promela.bin_expr#getOperand2 <em>Operand2</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getbin_expr()
 * @model
 * @generated
 */
public interface bin_expr extends any_expr {
	/**
	 * Returns the value of the '<em><b>Operator</b></em>' attribute.
	 * The literals are from the enumeration {@link promela.binarop}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operator</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operator</em>' attribute.
	 * @see promela.binarop
	 * @see #setOperator(binarop)
	 * @see promela.PromelaPackage#getbin_expr_Operator()
	 * @model required="true"
	 * @generated
	 */
	binarop getOperator();

	/**
	 * Sets the value of the '{@link promela.bin_expr#getOperator <em>Operator</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operator</em>' attribute.
	 * @see promela.binarop
	 * @see #getOperator()
	 * @generated
	 */
	void setOperator(binarop value);

	/**
	 * Returns the value of the '<em><b>Operand1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operand1</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operand1</em>' containment reference.
	 * @see #setOperand1(any_expr)
	 * @see promela.PromelaPackage#getbin_expr_Operand1()
	 * @model containment="true" required="true"
	 * @generated
	 */
	any_expr getOperand1();

	/**
	 * Sets the value of the '{@link promela.bin_expr#getOperand1 <em>Operand1</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operand1</em>' containment reference.
	 * @see #getOperand1()
	 * @generated
	 */
	void setOperand1(any_expr value);

	/**
	 * Returns the value of the '<em><b>Operand2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operand2</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operand2</em>' containment reference.
	 * @see #setOperand2(any_expr)
	 * @see promela.PromelaPackage#getbin_expr_Operand2()
	 * @model containment="true" required="true"
	 * @generated
	 */
	any_expr getOperand2();

	/**
	 * Sets the value of the '{@link promela.bin_expr#getOperand2 <em>Operand2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operand2</em>' containment reference.
	 * @see #getOperand2()
	 * @generated
	 */
	void setOperand2(any_expr value);

} // bin_expr
