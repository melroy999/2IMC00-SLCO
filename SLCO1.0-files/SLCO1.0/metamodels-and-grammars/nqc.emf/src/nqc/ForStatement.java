/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>For Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.ForStatement#getInitialization <em>Initialization</em>}</li>
 *   <li>{@link nqc.ForStatement#getCondition <em>Condition</em>}</li>
 *   <li>{@link nqc.ForStatement#getUpdate <em>Update</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getForStatement()
 * @model
 * @generated
 */
public interface ForStatement extends ControlStructure {
	/**
	 * Returns the value of the '<em><b>Initialization</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Initialization</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Initialization</em>' containment reference.
	 * @see #setInitialization(Statement)
	 * @see nqc.NqcPackage#getForStatement_Initialization()
	 * @model containment="true" ordered="false"
	 * @generated
	 */
	Statement getInitialization();

	/**
	 * Sets the value of the '{@link nqc.ForStatement#getInitialization <em>Initialization</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Initialization</em>' containment reference.
	 * @see #getInitialization()
	 * @generated
	 */
	void setInitialization(Statement value);

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
	 * @see nqc.NqcPackage#getForStatement_Condition()
	 * @model containment="true" ordered="false"
	 * @generated
	 */
	Expression getCondition();

	/**
	 * Sets the value of the '{@link nqc.ForStatement#getCondition <em>Condition</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Condition</em>' containment reference.
	 * @see #getCondition()
	 * @generated
	 */
	void setCondition(Expression value);

	/**
	 * Returns the value of the '<em><b>Update</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Update</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Update</em>' containment reference.
	 * @see #setUpdate(Statement)
	 * @see nqc.NqcPackage#getForStatement_Update()
	 * @model containment="true" ordered="false"
	 * @generated
	 */
	Statement getUpdate();

	/**
	 * Sets the value of the '{@link nqc.ForStatement#getUpdate <em>Update</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Update</em>' containment reference.
	 * @see #getUpdate()
	 * @generated
	 */
	void setUpdate(Statement value);

} // ForStatement
