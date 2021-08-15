/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc;

import nqc.DirectionEnum;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Signal2 Motor With Direction</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.Signal2MotorWithDirection#getOutputDirection <em>Output Direction</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getSignal2MotorWithDirection()
 * @model
 * @generated
 */
public interface Signal2MotorWithDirection extends Signal2Motor {
	/**
	 * Returns the value of the '<em><b>Output Direction</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.DirectionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Output Direction</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Output Direction</em>' attribute.
	 * @see nqc.DirectionEnum
	 * @see #setOutputDirection(DirectionEnum)
	 * @see slco2nqc.Slco2nqcPackage#getSignal2MotorWithDirection_OutputDirection()
	 * @model required="true"
	 * @generated
	 */
	DirectionEnum getOutputDirection();

	/**
	 * Sets the value of the '{@link slco2nqc.Signal2MotorWithDirection#getOutputDirection <em>Output Direction</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Output Direction</em>' attribute.
	 * @see nqc.DirectionEnum
	 * @see #getOutputDirection()
	 * @generated
	 */
	void setOutputDirection(DirectionEnum value);

} // Signal2MotorWithDirection
