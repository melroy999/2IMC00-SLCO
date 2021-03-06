/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.SerialPacketConstant;
import nqc.SerialPacketEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Serial Packet Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.SerialPacketConstantImpl#getSerialPacket <em>Serial Packet</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SerialPacketConstantImpl extends ConstantExpressionImpl implements SerialPacketConstant {
	/**
	 * The default value of the '{@link #getSerialPacket() <em>Serial Packet</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSerialPacket()
	 * @generated
	 * @ordered
	 */
	protected static final SerialPacketEnum SERIAL_PACKET_EDEFAULT = SerialPacketEnum.SERIAL_PACKET_DEFAULT;

	/**
	 * The cached value of the '{@link #getSerialPacket() <em>Serial Packet</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSerialPacket()
	 * @generated
	 * @ordered
	 */
	protected SerialPacketEnum serialPacket = SERIAL_PACKET_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SerialPacketConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getSerialPacketConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialPacketEnum getSerialPacket() {
		return serialPacket;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSerialPacket(SerialPacketEnum newSerialPacket) {
		SerialPacketEnum oldSerialPacket = serialPacket;
		serialPacket = newSerialPacket == null ? SERIAL_PACKET_EDEFAULT : newSerialPacket;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.SERIAL_PACKET_CONSTANT__SERIAL_PACKET, oldSerialPacket, serialPacket));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.SERIAL_PACKET_CONSTANT__SERIAL_PACKET:
				return getSerialPacket();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case NqcPackage.SERIAL_PACKET_CONSTANT__SERIAL_PACKET:
				setSerialPacket((SerialPacketEnum)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case NqcPackage.SERIAL_PACKET_CONSTANT__SERIAL_PACKET:
				setSerialPacket(SERIAL_PACKET_EDEFAULT);
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case NqcPackage.SERIAL_PACKET_CONSTANT__SERIAL_PACKET:
				return serialPacket != SERIAL_PACKET_EDEFAULT;
		}
		return super.eIsSet(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		if (eIsProxy()) return super.toString();

		StringBuffer result = new StringBuffer(super.toString());
		result.append(" (SerialPacket: ");
		result.append(serialPacket);
		result.append(')');
		return result.toString();
	}

} //SerialPacketConstantImpl
