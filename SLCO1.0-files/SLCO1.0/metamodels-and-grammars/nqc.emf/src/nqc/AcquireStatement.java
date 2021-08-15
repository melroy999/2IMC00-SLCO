/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Acquire Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.AcquireStatement#getResource <em>Resource</em>}</li>
 *   <li>{@link nqc.AcquireStatement#getBody <em>Body</em>}</li>
 *   <li>{@link nqc.AcquireStatement#getHandler <em>Handler</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getAcquireStatement()
 * @model
 * @generated
 */
public interface AcquireStatement extends Statement {
	/**
	 * Returns the value of the '<em><b>Resource</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Resource</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Resource</em>' containment reference.
	 * @see #setResource(AcquireStatement)
	 * @see nqc.NqcPackage#getAcquireStatement_Resource()
	 * @model containment="true" required="true"
	 * @generated
	 */
	AcquireStatement getResource();

	/**
	 * Sets the value of the '{@link nqc.AcquireStatement#getResource <em>Resource</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Resource</em>' containment reference.
	 * @see #getResource()
	 * @generated
	 */
	void setResource(AcquireStatement value);

	/**
	 * Returns the value of the '<em><b>Body</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Body</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Body</em>' containment reference.
	 * @see #setBody(Statement)
	 * @see nqc.NqcPackage#getAcquireStatement_Body()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Statement getBody();

	/**
	 * Sets the value of the '{@link nqc.AcquireStatement#getBody <em>Body</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Body</em>' containment reference.
	 * @see #getBody()
	 * @generated
	 */
	void setBody(Statement value);

	/**
	 * Returns the value of the '<em><b>Handler</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Handler</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Handler</em>' containment reference.
	 * @see #setHandler(Statement)
	 * @see nqc.NqcPackage#getAcquireStatement_Handler()
	 * @model containment="true"
	 * @generated
	 */
	Statement getHandler();

	/**
	 * Sets the value of the '{@link nqc.AcquireStatement#getHandler <em>Handler</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Handler</em>' containment reference.
	 * @see #getHandler()
	 * @generated
	 */
	void setHandler(Statement value);

} // AcquireStatement
