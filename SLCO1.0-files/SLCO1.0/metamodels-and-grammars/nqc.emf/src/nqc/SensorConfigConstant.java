/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Sensor Config Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SensorConfigConstant#getSensorConfig <em>Sensor Config</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSensorConfigConstant()
 * @model
 * @generated
 */
public interface SensorConfigConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Sensor Config</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SensorConfigEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sensor Config</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sensor Config</em>' attribute.
	 * @see nqc.SensorConfigEnum
	 * @see #setSensorConfig(SensorConfigEnum)
	 * @see nqc.NqcPackage#getSensorConfigConstant_SensorConfig()
	 * @model required="true"
	 * @generated
	 */
	SensorConfigEnum getSensorConfig();

	/**
	 * Sets the value of the '{@link nqc.SensorConfigConstant#getSensorConfig <em>Sensor Config</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sensor Config</em>' attribute.
	 * @see nqc.SensorConfigEnum
	 * @see #getSensorConfig()
	 * @generated
	 */
	void setSensorConfig(SensorConfigEnum value);

} // SensorConfigConstant
