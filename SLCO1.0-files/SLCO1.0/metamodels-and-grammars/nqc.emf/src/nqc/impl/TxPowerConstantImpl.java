/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.TxPowerConstant;
import nqc.TxPowerEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Tx Power Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.TxPowerConstantImpl#getTxPower <em>Tx Power</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class TxPowerConstantImpl extends ConstantExpressionImpl implements TxPowerConstant {
	/**
	 * The default value of the '{@link #getTxPower() <em>Tx Power</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTxPower()
	 * @generated
	 * @ordered
	 */
	protected static final TxPowerEnum TX_POWER_EDEFAULT = TxPowerEnum.TX_POWER_LO;

	/**
	 * The cached value of the '{@link #getTxPower() <em>Tx Power</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTxPower()
	 * @generated
	 * @ordered
	 */
	protected TxPowerEnum txPower = TX_POWER_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected TxPowerConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getTxPowerConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TxPowerEnum getTxPower() {
		return txPower;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setTxPower(TxPowerEnum newTxPower) {
		TxPowerEnum oldTxPower = txPower;
		txPower = newTxPower == null ? TX_POWER_EDEFAULT : newTxPower;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.TX_POWER_CONSTANT__TX_POWER, oldTxPower, txPower));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.TX_POWER_CONSTANT__TX_POWER:
				return getTxPower();
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
			case NqcPackage.TX_POWER_CONSTANT__TX_POWER:
				setTxPower((TxPowerEnum)newValue);
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
			case NqcPackage.TX_POWER_CONSTANT__TX_POWER:
				setTxPower(TX_POWER_EDEFAULT);
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
			case NqcPackage.TX_POWER_CONSTANT__TX_POWER:
				return txPower != TX_POWER_EDEFAULT;
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
		result.append(" (TxPower: ");
		result.append(txPower);
		result.append(')');
		return result.toString();
	}

} //TxPowerConstantImpl
