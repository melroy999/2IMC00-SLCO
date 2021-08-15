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
 * A representation of the model object '<em><b>Signal2 Sensor List</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.Signal2SensorList#getSignal2Sensors <em>Signal2 Sensors</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getSignal2SensorList()
 * @model
 * @generated
 */
public interface Signal2SensorList extends EObject {
	/**
	 * Returns the value of the '<em><b>Signal2 Sensors</b></em>' containment reference list.
	 * The list contents are of type {@link slco2nqc.Signal2Sensor}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Signal2 Sensors</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Signal2 Sensors</em>' containment reference list.
	 * @see slco2nqc.Slco2nqcPackage#getSignal2SensorList_Signal2Sensors()
	 * @model containment="true"
	 * @generated
	 */
	EList<Signal2Sensor> getSignal2Sensors();

} // Signal2SensorList
