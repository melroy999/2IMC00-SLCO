/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Ternary Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.TernaryExpression#getOperator <em>Operator</em>}</li>
 *   <li>{@link nqc.TernaryExpression#getOperand1 <em>Operand1</em>}</li>
 *   <li>{@link nqc.TernaryExpression#getOperand2 <em>Operand2</em>}</li>
 *   <li>{@link nqc.TernaryExpression#getOperand3 <em>Operand3</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getTernaryExpression()
 * @model
 * @generated
 */
public interface TernaryExpression extends CompoundExpression {
	/**
	 * Returns the value of the '<em><b>Operator</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.TernaryOperatorEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operator</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operator</em>' attribute.
	 * @see nqc.TernaryOperatorEnum
	 * @see #setOperator(TernaryOperatorEnum)
	 * @see nqc.NqcPackage#getTernaryExpression_Operator()
	 * @model
	 * @generated
	 */
	TernaryOperatorEnum getOperator();

	/**
	 * Sets the value of the '{@link nqc.TernaryExpression#getOperator <em>Operator</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operator</em>' attribute.
	 * @see nqc.TernaryOperatorEnum
	 * @see #getOperator()
	 * @generated
	 */
	void setOperator(TernaryOperatorEnum value);

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
	 * @see nqc.NqcPackage#getTernaryExpression_Operand1()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getOperand1();

	/**
	 * Sets the value of the '{@link nqc.TernaryExpression#getOperand1 <em>Operand1</em>}' containment reference.
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
	 * @see nqc.NqcPackage#getTernaryExpression_Operand2()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getOperand2();

	/**
	 * Sets the value of the '{@link nqc.TernaryExpression#getOperand2 <em>Operand2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operand2</em>' containment reference.
	 * @see #getOperand2()
	 * @generated
	 */
	void setOperand2(Expression value);

	/**
	 * Returns the value of the '<em><b>Operand3</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operand3</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operand3</em>' containment reference.
	 * @see #setOperand3(Expression)
	 * @see nqc.NqcPackage#getTernaryExpression_Operand3()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getOperand3();

	/**
	 * Sets the value of the '{@link nqc.TernaryExpression#getOperand3 <em>Operand3</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operand3</em>' containment reference.
	 * @see #getOperand3()
	 * @generated
	 */
	void setOperand3(Expression value);

} // TernaryExpression
