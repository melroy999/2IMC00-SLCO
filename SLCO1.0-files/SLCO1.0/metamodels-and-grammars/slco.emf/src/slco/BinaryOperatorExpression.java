/**
 */
package slco;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Binary Operator Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.BinaryOperatorExpression#getOperator <em>Operator</em>}</li>
 *   <li>{@link slco.BinaryOperatorExpression#getOperand1 <em>Operand1</em>}</li>
 *   <li>{@link slco.BinaryOperatorExpression#getOperand2 <em>Operand2</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getBinaryOperatorExpression()
 * @model
 * @generated
 */
public interface BinaryOperatorExpression extends Expression {
	/**
	 * Returns the value of the '<em><b>Operator</b></em>' attribute.
	 * The literals are from the enumeration {@link slco.OperatorEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operator</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operator</em>' attribute.
	 * @see slco.OperatorEnum
	 * @see #setOperator(OperatorEnum)
	 * @see slco.SlcoPackage#getBinaryOperatorExpression_Operator()
	 * @model required="true"
	 * @generated
	 */
	OperatorEnum getOperator();

	/**
	 * Sets the value of the '{@link slco.BinaryOperatorExpression#getOperator <em>Operator</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operator</em>' attribute.
	 * @see slco.OperatorEnum
	 * @see #getOperator()
	 * @generated
	 */
	void setOperator(OperatorEnum value);

	/**
	 * Returns the value of the '<em><b>Operand1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operand1</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operand1</em>' containment reference.
	 * @see #setOperand1(Expression)
	 * @see slco.SlcoPackage#getBinaryOperatorExpression_Operand1()
	 * @model containment="true" required="true" ordered="false"
	 * @generated
	 */
	Expression getOperand1();

	/**
	 * Sets the value of the '{@link slco.BinaryOperatorExpression#getOperand1 <em>Operand1</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operand1</em>' containment reference.
	 * @see #getOperand1()
	 * @generated
	 */
	void setOperand1(Expression value);

	/**
	 * Returns the value of the '<em><b>Operand2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operand2</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operand2</em>' containment reference.
	 * @see #setOperand2(Expression)
	 * @see slco.SlcoPackage#getBinaryOperatorExpression_Operand2()
	 * @model containment="true" required="true" ordered="false"
	 * @generated
	 */
	Expression getOperand2();

	/**
	 * Sets the value of the '{@link slco.BinaryOperatorExpression#getOperand2 <em>Operand2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operand2</em>' containment reference.
	 * @see #getOperand2()
	 * @generated
	 */
	void setOperand2(Expression value);

} // BinaryOperatorExpression
