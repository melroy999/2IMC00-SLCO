/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Serial Packet Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SerialPacketConstant#getSerialPacket <em>Serial Packet</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSerialPacketConstant()
 * @model
 * @generated
 */
public interface SerialPacketConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Serial Packet</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SerialPacketEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Serial Packet</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Serial Packet</em>' attribute.
	 * @see nqc.SerialPacketEnum
	 * @see #setSerialPacket(SerialPacketEnum)
	 * @see nqc.NqcPackage#getSerialPacketConstant_SerialPacket()
	 * @model required="true"
	 * @generated
	 */
	SerialPacketEnum getSerialPacket();

	/**
	 * Sets the value of the '{@link nqc.SerialPacketConstant#getSerialPacket <em>Serial Packet</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Serial Packet</em>' attribute.
	 * @see nqc.SerialPacketEnum
	 * @see #getSerialPacket()
	 * @generated
	 */
	void setSerialPacket(SerialPacketEnum value);

} // SerialPacketConstant
