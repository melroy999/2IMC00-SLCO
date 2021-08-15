/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.SerialChannelConstant;
import nqc.SerialChannelEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Serial Channel Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.SerialChannelConstantImpl#getSerialChannel <em>Serial Channel</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SerialChannelConstantImpl extends ConstantExpressionImpl implements SerialChannelConstant {
	/**
	 * The default value of the '{@link #getSerialChannel() <em>Serial Channel</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSerialChannel()
	 * @generated
	 * @ordered
	 */
	protected static final SerialChannelEnum SERIAL_CHANNEL_EDEFAULT = SerialChannelEnum.SERIAL_CHANNEL_IR;

	/**
	 * The cached value of the '{@link #getSerialChannel() <em>Serial Channel</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSerialChannel()
	 * @generated
	 * @ordered
	 */
	protected SerialChannelEnum serialChannel = SERIAL_CHANNEL_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SerialChannelConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getSerialChannelConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialChannelEnum getSerialChannel() {
		return serialChannel;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSerialChannel(SerialChannelEnum newSerialChannel) {
		SerialChannelEnum oldSerialChannel = serialChannel;
		serialChannel = newSerialChannel == null ? SERIAL_CHANNEL_EDEFAULT : newSerialChannel;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.SERIAL_CHANNEL_CONSTANT__SERIAL_CHANNEL, oldSerialChannel, serialChannel));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.SERIAL_CHANNEL_CONSTANT__SERIAL_CHANNEL:
				return getSerialChannel();
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
			case NqcPackage.SERIAL_CHANNEL_CONSTANT__SERIAL_CHANNEL:
				setSerialChannel((SerialChannelEnum)newValue);
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
			case NqcPackage.SERIAL_CHANNEL_CONSTANT__SERIAL_CHANNEL:
				setSerialChannel(SERIAL_CHANNEL_EDEFAULT);
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
			case NqcPackage.SERIAL_CHANNEL_CONSTANT__SERIAL_CHANNEL:
				return serialChannel != SERIAL_CHANNEL_EDEFAULT;
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
		result.append(" (SerialChannel: ");
		result.append(serialChannel);
		result.append(')');
		return result.toString();
	}

} //SerialChannelConstantImpl
