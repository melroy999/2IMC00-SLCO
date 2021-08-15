/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Unary Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.UnaryExpression#getOperand <em>Operand</em>}</li>
 *   <li>{@link nqc.UnaryExpression#getOperator <em>Operator</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getUnaryExpression()
 * @model
 * @generated
 */
public interface UnaryExpression extends CompoundExpression {
	/**
	 * Returns the value of the '<em><b>Operand</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operand</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operand</em>' containment reference.
	 * @see #setOperand(Expression)
	 * @see nqc.NqcPackage#getUnaryExpression_Operand()
	 * @model containment="true" required="true" ordered="false"
	 * @generated
	 */
	Expression getOperand();

	/**
	 * Sets the value of the '{@link nqc.UnaryExpression#getOperand <em>Operand</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operand</em>' containment reference.
	 * @see #getOperand()
	 * @generated
	 */
	void setOperand(Expression value);

	/**
	 * Returns the value of the '<em><b>Operator</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.UnaryOperatorEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operator</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operator</em>' attribute.
	 * @see nqc.UnaryOperatorEnum
	 * @see #setOperator(UnaryOperatorEnum)
	 * @see nqc.NqcPackage#getUnaryExpression_Operator()
	 * @model
	 * @generated
	 */
	UnaryOperatorEnum getOperator();

	/**
	 * Sets the value of the '{@link nqc.UnaryExpression#getOperator <em>Operator</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operator</em>' attribute.
	 * @see nqc.UnaryOperatorEnum
	 * @see #getOperator()
	 * @generated
	 */
	void setOperator(UnaryOperatorEnum value);

} // UnaryExpression
