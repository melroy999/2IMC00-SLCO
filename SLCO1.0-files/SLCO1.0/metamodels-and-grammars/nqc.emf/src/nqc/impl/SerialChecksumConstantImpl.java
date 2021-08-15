/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.SerialChecksumConstant;
import nqc.SerialChecksumEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Serial Checksum Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.SerialChecksumConstantImpl#getSerialChecksum <em>Serial Checksum</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SerialChecksumConstantImpl extends ConstantExpressionImpl implements SerialChecksumConstant {
	/**
	 * The default value of the '{@link #getSerialChecksum() <em>Serial Checksum</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSerialChecksum()
	 * @generated
	 * @ordered
	 */
	protected static final SerialChecksumEnum SERIAL_CHECKSUM_EDEFAULT = SerialChecksumEnum.SERIAL_CHECKSUM_NONE;

	/**
	 * The cached value of the '{@link #getSerialChecksum() <em>Serial Checksum</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSerialChecksum()
	 * @generated
	 * @ordered
	 */
	protected SerialChecksumEnum serialChecksum = SERIAL_CHECKSUM_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SerialChecksumConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getSerialChecksumConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialChecksumEnum getSerialChecksum() {
		return serialChecksum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSerialChecksum(SerialChecksumEnum newSerialChecksum) {
		SerialChecksumEnum oldSerialChecksum = serialChecksum;
		serialChecksum = newSerialChecksum == null ? SERIAL_CHECKSUM_EDEFAULT : newSerialChecksum;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.SERIAL_CHECKSUM_CONSTANT__SERIAL_CHECKSUM, oldSerialChecksum, serialChecksum));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.SERIAL_CHECKSUM_CONSTANT__SERIAL_CHECKSUM:
				return getSerialChecksum();
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
			case NqcPackage.SERIAL_CHECKSUM_CONSTANT__SERIAL_CHECKSUM:
				setSerialChecksum((SerialChecksumEnum)newValue);
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
			case NqcPackage.SERIAL_CHECKSUM_CONSTANT__SERIAL_CHECKSUM:
				setSerialChecksum(SERIAL_CHECKSUM_EDEFAULT);
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
			case NqcPackage.SERIAL_CHECKSUM_CONSTANT__SERIAL_CHECKSUM:
				return serialChecksum != SERIAL_CHECKSUM_EDEFAULT;
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
		result.append(" (SerialChecksum: ");
		result.append(serialChecksum);
		result.append(')');
		return result.toString();
	}

} //SerialChecksumConstantImpl
