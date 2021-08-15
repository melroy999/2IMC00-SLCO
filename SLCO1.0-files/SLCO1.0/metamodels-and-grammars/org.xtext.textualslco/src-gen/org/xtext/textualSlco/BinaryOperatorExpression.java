/**
 */
package org.xtext.textualSlco;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Binary Operator Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.BinaryOperatorExpression#getOperand1 <em>Operand1</em>}</li>
 *   <li>{@link org.xtext.textualSlco.BinaryOperatorExpression#getOperator <em>Operator</em>}</li>
 *   <li>{@link org.xtext.textualSlco.BinaryOperatorExpression#getOperand2 <em>Operand2</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getBinaryOperatorExpression()
 * @model
 * @generated
 */
public interface BinaryOperatorExpression extends Expression
{
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
   * @see org.xtext.textualSlco.TextualSlcoPackage#getBinaryOperatorExpression_Operand1()
   * @model containment="true"
   * @generated
   */
  Expression getOperand1();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.BinaryOperatorExpression#getOperand1 <em>Operand1</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Operand1</em>' containment reference.
   * @see #getOperand1()
   * @generated
   */
  void setOperand1(Expression value);

  /**
   * Returns the value of the '<em><b>Operator</b></em>' attribute.
   * The literals are from the enumeration {@link org.xtext.textualSlco.Operator}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Operator</em>' attribute isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Operator</em>' attribute.
   * @see org.xtext.textualSlco.Operator
   * @see #setOperator(Operator)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getBinaryOperatorExpression_Operator()
   * @model
   * @generated
   */
  Operator getOperator();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.BinaryOperatorExpression#getOperator <em>Operator</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Operator</em>' attribute.
   * @see org.xtext.textualSlco.Operator
   * @see #getOperator()
   * @generated
   */
  void setOperator(Operator value);

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
   * @see org.xtext.textualSlco.TextualSlcoPackage#getBinaryOperatorExpression_Operand2()
   * @model containment="true"
   * @generated
   */
  Expression getOperand2();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.BinaryOperatorExpression#getOperand2 <em>Operand2</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Operand2</em>' containment reference.
   * @see #getOperand2()
   * @generated
   */
  void setOperand2(Expression value);

} // BinaryOperatorExpression
