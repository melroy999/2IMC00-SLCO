/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Slco2 Nqc</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.Slco2Nqc#getSignal2MotorList <em>Signal2 Motor List</em>}</li>
 *   <li>{@link slco2nqc.Slco2Nqc#getSignal2SensorList <em>Signal2 Sensor List</em>}</li>
 *   <li>{@link slco2nqc.Slco2Nqc#getMotorInitList <em>Motor Init List</em>}</li>
 *   <li>{@link slco2nqc.Slco2Nqc#getClassList <em>Class List</em>}</li>
 *   <li>{@link slco2nqc.Slco2Nqc#getSensorInitList <em>Sensor Init List</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getSlco2Nqc()
 * @model
 * @generated
 */
public interface Slco2Nqc extends EObject {
	/**
	 * Returns the value of the '<em><b>Signal2 Motor List</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Signal2 Motor List</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Signal2 Motor List</em>' containment reference.
	 * @see #setSignal2MotorList(Signal2MotorList)
	 * @see slco2nqc.Slco2nqcPackage#getSlco2Nqc_Signal2MotorList()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Signal2MotorList getSignal2MotorList();

	/**
	 * Sets the value of the '{@link slco2nqc.Slco2Nqc#getSignal2MotorList <em>Signal2 Motor List</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Signal2 Motor List</em>' containment reference.
	 * @see #getSignal2MotorList()
	 * @generated
	 */
	void setSignal2MotorList(Signal2MotorList value);

	/**
	 * Returns the value of the '<em><b>Signal2 Sensor List</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Signal2 Sensor List</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Signal2 Sensor List</em>' containment reference.
	 * @see #setSignal2SensorList(Signal2SensorList)
	 * @see slco2nqc.Slco2nqcPackage#getSlco2Nqc_Signal2SensorList()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Signal2SensorList getSignal2SensorList();

	/**
	 * Sets the value of the '{@link slco2nqc.Slco2Nqc#getSignal2SensorList <em>Signal2 Sensor List</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Signal2 Sensor List</em>' containment reference.
	 * @see #getSignal2SensorList()
	 * @generated
	 */
	void setSignal2SensorList(Signal2SensorList value);

	/**
	 * Returns the value of the '<em><b>Motor Init List</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Motor Init List</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Motor Init List</em>' containment reference.
	 * @see #setMotorInitList(MotorInitList)
	 * @see slco2nqc.Slco2nqcPackage#getSlco2Nqc_MotorInitList()
	 * @model containment="true" required="true"
	 * @generated
	 */
	MotorInitList getMotorInitList();

	/**
	 * Sets the value of the '{@link slco2nqc.Slco2Nqc#getMotorInitList <em>Motor Init List</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Motor Init List</em>' containment reference.
	 * @see #getMotorInitList()
	 * @generated
	 */
	void setMotorInitList(MotorInitList value);

	/**
	 * Returns the value of the '<em><b>Class List</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Class List</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Class List</em>' containment reference.
	 * @see #setClassList(ClassList)
	 * @see slco2nqc.Slco2nqcPackage#getSlco2Nqc_ClassList()
	 * @model containment="true" required="true"
	 * @generated
	 */
	ClassList getClassList();

	/**
	 * Sets the value of the '{@link slco2nqc.Slco2Nqc#getClassList <em>Class List</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Class List</em>' containment reference.
	 * @see #getClassList()
	 * @generated
	 */
	void setClassList(ClassList value);

	/**
	 * Returns the value of the '<em><b>Sensor Init List</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sensor Init List</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sensor Init List</em>' containment reference.
	 * @see #setSensorInitList(SensorInitList)
	 * @see slco2nqc.Slco2nqcPackage#getSlco2Nqc_SensorInitList()
	 * @model containment="true" required="true"
	 * @generated
	 */
	SensorInitList getSensorInitList();

	/**
	 * Sets the value of the '{@link slco2nqc.Slco2Nqc#getSensorInitList <em>Sensor Init List</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sensor Init List</em>' containment reference.
	 * @see #getSensorInitList()
	 * @generated
	 */
	void setSensorInitList(SensorInitList value);

} // Slco2Nqc
