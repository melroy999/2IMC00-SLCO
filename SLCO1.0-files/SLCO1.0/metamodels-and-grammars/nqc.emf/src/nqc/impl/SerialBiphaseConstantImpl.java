/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.SerialBiphaseConstant;
import nqc.SerialBiphaseEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Serial Biphase Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.SerialBiphaseConstantImpl#getSerialBiphase <em>Serial Biphase</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SerialBiphaseConstantImpl extends ConstantExpressionImpl implements SerialBiphaseConstant {
	/**
	 * The default value of the '{@link #getSerialBiphase() <em>Serial Biphase</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSerialBiphase()
	 * @generated
	 * @ordered
	 */
	protected static final SerialBiphaseEnum SERIAL_BIPHASE_EDEFAULT = SerialBiphaseEnum.SERIAL_BIPHASE_OFF;

	/**
	 * The cached value of the '{@link #getSerialBiphase() <em>Serial Biphase</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSerialBiphase()
	 * @generated
	 * @ordered
	 */
	protected SerialBiphaseEnum serialBiphase = SERIAL_BIPHASE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SerialBiphaseConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getSerialBiphaseConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialBiphaseEnum getSerialBiphase() {
		return serialBiphase;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSerialBiphase(SerialBiphaseEnum newSerialBiphase) {
		SerialBiphaseEnum oldSerialBiphase = serialBiphase;
		serialBiphase = newSerialBiphase == null ? SERIAL_BIPHASE_EDEFAULT : newSerialBiphase;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.SERIAL_BIPHASE_CONSTANT__SERIAL_BIPHASE, oldSerialBiphase, serialBiphase));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.SERIAL_BIPHASE_CONSTANT__SERIAL_BIPHASE:
				return getSerialBiphase();
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
			case NqcPackage.SERIAL_BIPHASE_CONSTANT__SERIAL_BIPHASE:
				setSerialBiphase((SerialBiphaseEnum)newValue);
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
			case NqcPackage.SERIAL_BIPHASE_CONSTANT__SERIAL_BIPHASE:
				setSerialBiphase(SERIAL_BIPHASE_EDEFAULT);
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
			case NqcPackage.SERIAL_BIPHASE_CONSTANT__SERIAL_BIPHASE:
				return serialBiphase != SERIAL_BIPHASE_EDEFAULT;
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
		result.append(" (SerialBiphase: ");
		result.append(serialBiphase);
		result.append(')');
		return result.toString();
	}

} //SerialBiphaseConstantImpl
