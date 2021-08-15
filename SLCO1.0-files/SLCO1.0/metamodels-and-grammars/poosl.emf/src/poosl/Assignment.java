/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Assignment</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.Assignment#getRightHandSide <em>Right Hand Side</em>}</li>
 *   <li>{@link poosl.Assignment#getLeftHandSide <em>Left Hand Side</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getAssignment()
 * @model
 * @generated
 */
public interface Assignment extends Expression {
	/**
	 * Returns the value of the '<em><b>Right Hand Side</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Right Hand Side</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Right Hand Side</em>' containment reference list.
	 * @see poosl.PooslPackage#getAssignment_RightHandSide()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Expression> getRightHandSide();

	/**
	 * Returns the value of the '<em><b>Left Hand Side</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Left Hand Side</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Left Hand Side</em>' reference.
	 * @see #setLeftHandSide(Variable)
	 * @see poosl.PooslPackage#getAssignment_LeftHandSide()
	 * @model required="true"
	 * @generated
	 */
	Variable getLeftHandSide();

	/**
	 * Sets the value of the '{@link poosl.Assignment#getLeftHandSide <em>Left Hand Side</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Left Hand Side</em>' reference.
	 * @see #getLeftHandSide()
	 * @generated
	 */
	void setLeftHandSide(Variable value);

} // Assignment
