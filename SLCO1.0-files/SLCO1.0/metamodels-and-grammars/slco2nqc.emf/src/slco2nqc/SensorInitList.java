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
 * A representation of the model object '<em><b>Sensor Init List</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.SensorInitList#getSensorInits <em>Sensor Inits</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getSensorInitList()
 * @model
 * @generated
 */
public interface SensorInitList extends EObject {
	/**
	 * Returns the value of the '<em><b>Sensor Inits</b></em>' containment reference list.
	 * The list contents are of type {@link slco2nqc.SensorInit}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sensor Inits</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sensor Inits</em>' containment reference list.
	 * @see slco2nqc.Slco2nqcPackage#getSensorInitList_SensorInits()
	 * @model containment="true"
	 * @generated
	 */
	EList<SensorInit> getSensorInits();

} // SensorInitList
