/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Subroutine Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SubroutineCall#getCallee <em>Callee</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSubroutineCall()
 * @model
 * @generated
 */
public interface SubroutineCall extends CallStatement {
	/**
	 * Returns the value of the '<em><b>Callee</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Callee</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Callee</em>' reference.
	 * @see #setCallee(Subroutine)
	 * @see nqc.NqcPackage#getSubroutineCall_Callee()
	 * @model required="true" ordered="false"
	 * @generated
	 */
	Subroutine getCallee();

	/**
	 * Sets the value of the '{@link nqc.SubroutineCall#getCallee <em>Callee</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Callee</em>' reference.
	 * @see #getCallee()
	 * @generated
	 */
	void setCallee(Subroutine value);

} // SubroutineCall
