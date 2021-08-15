/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Sensor Mode Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SensorModeConstant#getSensorMode <em>Sensor Mode</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSensorModeConstant()
 * @model
 * @generated
 */
public interface SensorModeConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Sensor Mode</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SensorModeEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sensor Mode</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sensor Mode</em>' attribute.
	 * @see nqc.SensorModeEnum
	 * @see #setSensorMode(SensorModeEnum)
	 * @see nqc.NqcPackage#getSensorModeConstant_SensorMode()
	 * @model required="true"
	 * @generated
	 */
	SensorModeEnum getSensorMode();

	/**
	 * Sets the value of the '{@link nqc.SensorModeConstant#getSensorMode <em>Sensor Mode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sensor Mode</em>' attribute.
	 * @see nqc.SensorModeEnum
	 * @see #getSensorMode()
	 * @generated
	 */
	void setSensorMode(SensorModeEnum value);

} // SensorModeConstant
