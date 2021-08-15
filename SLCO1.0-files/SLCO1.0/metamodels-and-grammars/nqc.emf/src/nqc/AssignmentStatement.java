/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Assignment Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.AssignmentStatement#getVariable <em>Variable</em>}</li>
 *   <li>{@link nqc.AssignmentStatement#getExpression <em>Expression</em>}</li>
 *   <li>{@link nqc.AssignmentStatement#getOperator <em>Operator</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getAssignmentStatement()
 * @model
 * @generated
 */
public interface AssignmentStatement extends Statement {
	/**
	 * Returns the value of the '<em><b>Variable</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Variable</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Variable</em>' containment reference.
	 * @see #setVariable(VariableExpression)
	 * @see nqc.NqcPackage#getAssignmentStatement_Variable()
	 * @model containment="true" required="true" ordered="false"
	 * @generated
	 */
	VariableExpression getVariable();

	/**
	 * Sets the value of the '{@link nqc.AssignmentStatement#getVariable <em>Variable</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Variable</em>' containment reference.
	 * @see #getVariable()
	 * @generated
	 */
	void setVariable(VariableExpression value);

	/**
	 * Returns the value of the '<em><b>Expression</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Expression</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Expression</em>' containment reference.
	 * @see #setExpression(Expression)
	 * @see nqc.NqcPackage#getAssignmentStatement_Expression()
	 * @model containment="true" required="true" ordered="false"
	 * @generated
	 */
	Expression getExpression();

	/**
	 * Sets the value of the '{@link nqc.AssignmentStatement#getExpression <em>Expression</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Expression</em>' containment reference.
	 * @see #getExpression()
	 * @generated
	 */
	void setExpression(Expression value);

	/**
	 * Returns the value of the '<em><b>Operator</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.AssignmentStatementEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Operator</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Operator</em>' attribute.
	 * @see nqc.AssignmentStatementEnum
	 * @see #setOperator(AssignmentStatementEnum)
	 * @see nqc.NqcPackage#getAssignmentStatement_Operator()
	 * @model
	 * @generated
	 */
	AssignmentStatementEnum getOperator();

	/**
	 * Sets the value of the '{@link nqc.AssignmentStatement#getOperator <em>Operator</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Operator</em>' attribute.
	 * @see nqc.AssignmentStatementEnum
	 * @see #getOperator()
	 * @generated
	 */
	void setOperator(AssignmentStatementEnum value);

} // AssignmentStatement
