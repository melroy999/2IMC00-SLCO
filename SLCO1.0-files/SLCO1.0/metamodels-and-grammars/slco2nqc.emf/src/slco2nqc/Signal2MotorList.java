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
 * A representation of the model object '<em><b>Signal2 Motor List</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.Signal2MotorList#getSignal2Motors <em>Signal2 Motors</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getSignal2MotorList()
 * @model
 * @generated
 */
public interface Signal2MotorList extends EObject {
	/**
	 * Returns the value of the '<em><b>Signal2 Motors</b></em>' containment reference list.
	 * The list contents are of type {@link slco2nqc.Signal2Motor}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Signal2 Motors</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Signal2 Motors</em>' containment reference list.
	 * @see slco2nqc.Slco2nqcPackage#getSignal2MotorList_Signal2Motors()
	 * @model containment="true"
	 * @generated
	 */
	EList<Signal2Motor> getSignal2Motors();

} // Signal2MotorList
