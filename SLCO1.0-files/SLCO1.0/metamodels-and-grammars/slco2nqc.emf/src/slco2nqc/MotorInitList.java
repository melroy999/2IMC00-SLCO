/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Motor Init List</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.MotorInitList#getMotorInits <em>Motor Inits</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getMotorInitList()
 * @model
 * @generated
 */
public interface MotorInitList extends EObject {
	/**
	 * Returns the value of the '<em><b>Motor Inits</b></em>' containment reference list.
	 * The list contents are of type {@link slco2nqc.MotorInit}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Motor Inits</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Motor Inits</em>' containment reference list.
	 * @see slco2nqc.Slco2nqcPackage#getMotorInitList_MotorInits()
	 * @model containment="true"
	 * @generated
	 */
	EList<MotorInit> getMotorInits();

} // MotorInitList
