/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.AcquireConstant;
import nqc.AcquireEnum;
import nqc.NqcPackage;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Acquire Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.AcquireConstantImpl#getAcquire <em>Acquire</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class AcquireConstantImpl extends ConstantExpressionImpl implements AcquireConstant {
	/**
	 * The default value of the '{@link #getAcquire() <em>Acquire</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAcquire()
	 * @generated
	 * @ordered
	 */
	protected static final AcquireEnum ACQUIRE_EDEFAULT = AcquireEnum.ACQUIRE_OUT_A;

	/**
	 * The cached value of the '{@link #getAcquire() <em>Acquire</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAcquire()
	 * @generated
	 * @ordered
	 */
	protected AcquireEnum acquire = ACQUIRE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected AcquireConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getAcquireConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AcquireEnum getAcquire() {
		return acquire;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setAcquire(AcquireEnum newAcquire) {
		AcquireEnum oldAcquire = acquire;
		acquire = newAcquire == null ? ACQUIRE_EDEFAULT : newAcquire;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.ACQUIRE_CONSTANT__ACQUIRE, oldAcquire, acquire));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.ACQUIRE_CONSTANT__ACQUIRE:
				return getAcquire();
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
			case NqcPackage.ACQUIRE_CONSTANT__ACQUIRE:
				setAcquire((AcquireEnum)newValue);
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
			case NqcPackage.ACQUIRE_CONSTANT__ACQUIRE:
				setAcquire(ACQUIRE_EDEFAULT);
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
			case NqcPackage.ACQUIRE_CONSTANT__ACQUIRE:
				return acquire != ACQUIRE_EDEFAULT;
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
		result.append(" (Acquire: ");
		result.append(acquire);
		result.append(')');
		return result.toString();
	}

} //AcquireConstantImpl
