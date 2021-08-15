/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.SerialCommConstant;
import nqc.SerialCommEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Serial Comm Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.SerialCommConstantImpl#getSerialComm <em>Serial Comm</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SerialCommConstantImpl extends ConstantExpressionImpl implements SerialCommConstant {
	/**
	 * The default value of the '{@link #getSerialComm() <em>Serial Comm</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSerialComm()
	 * @generated
	 * @ordered
	 */
	protected static final SerialCommEnum SERIAL_COMM_EDEFAULT = SerialCommEnum.SERIAL_COMM_DEFAULT;

	/**
	 * The cached value of the '{@link #getSerialComm() <em>Serial Comm</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSerialComm()
	 * @generated
	 * @ordered
	 */
	protected SerialCommEnum serialComm = SERIAL_COMM_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SerialCommConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getSerialCommConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialCommEnum getSerialComm() {
		return serialComm;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSerialComm(SerialCommEnum newSerialComm) {
		SerialCommEnum oldSerialComm = serialComm;
		serialComm = newSerialComm == null ? SERIAL_COMM_EDEFAULT : newSerialComm;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.SERIAL_COMM_CONSTANT__SERIAL_COMM, oldSerialComm, serialComm));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.SERIAL_COMM_CONSTANT__SERIAL_COMM:
				return getSerialComm();
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
			case NqcPackage.SERIAL_COMM_CONSTANT__SERIAL_COMM:
				setSerialComm((SerialCommEnum)newValue);
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
			case NqcPackage.SERIAL_COMM_CONSTANT__SERIAL_COMM:
				setSerialComm(SERIAL_COMM_EDEFAULT);
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
			case NqcPackage.SERIAL_COMM_CONSTANT__SERIAL_COMM:
				return serialComm != SERIAL_COMM_EDEFAULT;
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
		result.append(" (SerialComm: ");
		result.append(serialComm);
		result.append(')');
		return result.toString();
	}

} //SerialCommConstantImpl
