/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Go To Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.GoToStatement#getJumpLabel <em>Jump Label</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getGoToStatement()
 * @model
 * @generated
 */
public interface GoToStatement extends ControlStructure {
	/**
	 * Returns the value of the '<em><b>Jump Label</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Jump Label</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Jump Label</em>' reference.
	 * @see #setJumpLabel(Label)
	 * @see nqc.NqcPackage#getGoToStatement_JumpLabel()
	 * @model required="true" ordered="false"
	 * @generated
	 */
	Label getJumpLabel();

	/**
	 * Sets the value of the '{@link nqc.GoToStatement#getJumpLabel <em>Jump Label</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Jump Label</em>' reference.
	 * @see #getJumpLabel()
	 * @generated
	 */
	void setJumpLabel(Label value);

} // GoToStatement
