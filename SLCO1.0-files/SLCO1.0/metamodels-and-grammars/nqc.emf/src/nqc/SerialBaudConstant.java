/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Serial Baud Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SerialBaudConstant#getSerialBaud <em>Serial Baud</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSerialBaudConstant()
 * @model
 * @generated
 */
public interface SerialBaudConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Serial Baud</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SerialBaudEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Serial Baud</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Serial Baud</em>' attribute.
	 * @see nqc.SerialBaudEnum
	 * @see #setSerialBaud(SerialBaudEnum)
	 * @see nqc.NqcPackage#getSerialBaudConstant_SerialBaud()
	 * @model required="true"
	 * @generated
	 */
	SerialBaudEnum getSerialBaud();

	/**
	 * Sets the value of the '{@link nqc.SerialBaudConstant#getSerialBaud <em>Serial Baud</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Serial Baud</em>' attribute.
	 * @see nqc.SerialBaudEnum
	 * @see #getSerialBaud()
	 * @generated
	 */
	void setSerialBaud(SerialBaudEnum value);

} // SerialBaudConstant
