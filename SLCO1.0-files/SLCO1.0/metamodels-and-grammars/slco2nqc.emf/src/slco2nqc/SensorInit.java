/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc;

import nqc.SensorConfigEnum;
import nqc.SensorNameEnum;

import org.eclipse.emf.ecore.EObject;

import slco.Port;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Sensor Init</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.SensorInit#getSensorName <em>Sensor Name</em>}</li>
 *   <li>{@link slco2nqc.SensorInit#getSensorConfiguration <em>Sensor Configuration</em>}</li>
 *   <li>{@link slco2nqc.SensorInit#getPort <em>Port</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getSensorInit()
 * @model
 * @generated
 */
public interface SensorInit extends EObject {
	/**
	 * Returns the value of the '<em><b>Sensor Name</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SensorNameEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sensor Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sensor Name</em>' attribute.
	 * @see nqc.SensorNameEnum
	 * @see #setSensorName(SensorNameEnum)
	 * @see slco2nqc.Slco2nqcPackage#getSensorInit_SensorName()
	 * @model required="true"
	 * @generated
	 */
	SensorNameEnum getSensorName();

	/**
	 * Sets the value of the '{@link slco2nqc.SensorInit#getSensorName <em>Sensor Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sensor Name</em>' attribute.
	 * @see nqc.SensorNameEnum
	 * @see #getSensorName()
	 * @generated
	 */
	void setSensorName(SensorNameEnum value);

	/**
	 * Returns the value of the '<em><b>Sensor Configuration</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SensorConfigEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sensor Configuration</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sensor Configuration</em>' attribute.
	 * @see nqc.SensorConfigEnum
	 * @see #setSensorConfiguration(SensorConfigEnum)
	 * @see slco2nqc.Slco2nqcPackage#getSensorInit_SensorConfiguration()
	 * @model required="true"
	 * @generated
	 */
	SensorConfigEnum getSensorConfiguration();

	/**
	 * Sets the value of the '{@link slco2nqc.SensorInit#getSensorConfiguration <em>Sensor Configuration</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sensor Configuration</em>' attribute.
	 * @see nqc.SensorConfigEnum
	 * @see #getSensorConfiguration()
	 * @generated
	 */
	void setSensorConfiguration(SensorConfigEnum value);

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
	 * @see slco2nqc.Slco2nqcPackage#getSensorInit_Port()
	 * @model required="true"
	 * @generated
	 */
	Port getPort();

	/**
	 * Sets the value of the '{@link slco2nqc.SensorInit#getPort <em>Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Port</em>' reference.
	 * @see #getPort()
	 * @generated
	 */
	void setPort(Port value);

} // SensorInit
