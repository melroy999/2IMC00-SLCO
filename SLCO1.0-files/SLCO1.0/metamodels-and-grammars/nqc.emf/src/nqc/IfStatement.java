/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>If Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.IfStatement#getCondition <em>Condition</em>}</li>
 *   <li>{@link nqc.IfStatement#getConsequence <em>Consequence</em>}</li>
 *   <li>{@link nqc.IfStatement#getAlternative <em>Alternative</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getIfStatement()
 * @model
 * @generated
 */
public interface IfStatement extends ControlStructure {
	/**
	 * Returns the value of the '<em><b>Condition</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Condition</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Condition</em>' containment reference.
	 * @see #setCondition(Expression)
	 * @see nqc.NqcPackage#getIfStatement_Condition()
	 * @model containment="true" required="true" ordered="false"
	 * @generated
	 */
	Expression getCondition();

	/**
	 * Sets the value of the '{@link nqc.IfStatement#getCondition <em>Condition</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Condition</em>' containment reference.
	 * @see #getCondition()
	 * @generated
	 */
	void setCondition(Expression value);

	/**
	 * Returns the value of the '<em><b>Consequence</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Consequence</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Consequence</em>' containment reference.
	 * @see #setConsequence(Statement)
	 * @see nqc.NqcPackage#getIfStatement_Consequence()
	 * @model containment="true" required="true" ordered="false"
	 * @generated
	 */
	Statement getConsequence();

	/**
	 * Sets the value of the '{@link nqc.IfStatement#getConsequence <em>Consequence</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Consequence</em>' containment reference.
	 * @see #getConsequence()
	 * @generated
	 */
	void setConsequence(Statement value);

	/**
	 * Returns the value of the '<em><b>Alternative</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Alternative</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Alternative</em>' containment reference.
	 * @see #setAlternative(Statement)
	 * @see nqc.NqcPackage#getIfStatement_Alternative()
	 * @model containment="true" ordered="false"
	 * @generated
	 */
	Statement getAlternative();

	/**
	 * Sets the value of the '{@link nqc.IfStatement#getAlternative <em>Alternative</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Alternative</em>' containment reference.
	 * @see #getAlternative()
	 * @generated
	 */
	void setAlternative(Statement value);

} // IfStatement
