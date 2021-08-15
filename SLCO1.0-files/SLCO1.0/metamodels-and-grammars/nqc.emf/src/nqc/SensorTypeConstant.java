/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Sensor Type Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SensorTypeConstant#getSensorType <em>Sensor Type</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSensorTypeConstant()
 * @model
 * @generated
 */
public interface SensorTypeConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Sensor Type</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SensorTypeEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sensor Type</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sensor Type</em>' attribute.
	 * @see nqc.SensorTypeEnum
	 * @see #setSensorType(SensorTypeEnum)
	 * @see nqc.NqcPackage#getSensorTypeConstant_SensorType()
	 * @model required="true"
	 * @generated
	 */
	SensorTypeEnum getSensorType();

	/**
	 * Sets the value of the '{@link nqc.SensorTypeConstant#getSensorType <em>Sensor Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sensor Type</em>' attribute.
	 * @see nqc.SensorTypeEnum
	 * @see #getSensorType()
	 * @generated
	 */
	void setSensorType(SensorTypeEnum value);

} // SensorTypeConstant
