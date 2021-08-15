/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Sensor Name Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SensorNameConstant#getSensorName <em>Sensor Name</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSensorNameConstant()
 * @model
 * @generated
 */
public interface SensorNameConstant extends ConstantExpression {
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
	 * @see nqc.NqcPackage#getSensorNameConstant_SensorName()
	 * @model required="true"
	 * @generated
	 */
	SensorNameEnum getSensorName();

	/**
	 * Sets the value of the '{@link nqc.SensorNameConstant#getSensorName <em>Sensor Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sensor Name</em>' attribute.
	 * @see nqc.SensorNameEnum
	 * @see #getSensorName()
	 * @generated
	 */
	void setSensorName(SensorNameEnum value);

} // SensorNameConstant
