/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc;

import org.eclipse.emf.ecore.EObject;

import slco.Port;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Signal2 Sensor</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.Signal2Sensor#getSignalName <em>Signal Name</em>}</li>
 *   <li>{@link slco2nqc.Signal2Sensor#getPort <em>Port</em>}</li>
 *   <li>{@link slco2nqc.Signal2Sensor#isSensorIsHigh <em>Sensor Is High</em>}</li>
 *   <li>{@link slco2nqc.Signal2Sensor#getTreshold <em>Treshold</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getSignal2Sensor()
 * @model
 * @generated
 */
public interface Signal2Sensor extends EObject {
	/**
	 * Returns the value of the '<em><b>Signal Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Signal Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Signal Name</em>' attribute.
	 * @see #setSignalName(String)
	 * @see slco2nqc.Slco2nqcPackage#getSignal2Sensor_SignalName()
	 * @model required="true"
	 * @generated
	 */
	String getSignalName();

	/**
	 * Sets the value of the '{@link slco2nqc.Signal2Sensor#getSignalName <em>Signal Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Signal Name</em>' attribute.
	 * @see #getSignalName()
	 * @generated
	 */
	void setSignalName(String value);

	/**
	 * Returns the value of the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Port</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Port</em>' reference.
	 * @see #setPort(Port)
	 * @see slco2nqc.Slco2nqcPackage#getSignal2Sensor_Port()
	 * @model required="true"
	 * @generated
	 */
	Port getPort();

	/**
	 * Sets the value of the '{@link slco2nqc.Signal2Sensor#getPort <em>Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Port</em>' reference.
	 * @see #getPort()
	 * @generated
	 */
	void setPort(Port value);

	/**
	 * Returns the value of the '<em><b>Sensor Is High</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sensor Is High</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sensor Is High</em>' attribute.
	 * @see #setSensorIsHigh(boolean)
	 * @see slco2nqc.Slco2nqcPackage#getSignal2Sensor_SensorIsHigh()
	 * @model required="true"
	 * @generated
	 */
	boolean isSensorIsHigh();

	/**
	 * Sets the value of the '{@link slco2nqc.Signal2Sensor#isSensorIsHigh <em>Sensor Is High</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sensor Is High</em>' attribute.
	 * @see #isSensorIsHigh()
	 * @generated
	 */
	void setSensorIsHigh(boolean value);

	/**
	 * Returns the value of the '<em><b>Treshold</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Treshold</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Treshold</em>' attribute.
	 * @see #setTreshold(int)
	 * @see slco2nqc.Slco2nqcPackage#getSignal2Sensor_Treshold()
	 * @model
	 * @generated
	 */
	int getTreshold();

	/**
	 * Sets the value of the '{@link slco2nqc.Signal2Sensor#getTreshold <em>Treshold</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Treshold</em>' attribute.
	 * @see #getTreshold()
	 * @generated
	 */
	void setTreshold(int value);

} // Signal2Sensor
