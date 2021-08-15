/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import promela.PromelaPackage;
import promela.mtype;
import promela.mtype_const;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>mtype const</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.mtype_constImpl#getMtype <em>Mtype</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class mtype_constImpl extends constantImpl implements mtype_const {
	/**
	 * The cached value of the '{@link #getMtype() <em>Mtype</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMtype()
	 * @generated
	 * @ordered
	 */
	protected mtype mtype;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected mtype_constImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.MTYPE_CONST;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public mtype getMtype() {
		if (mtype != null && mtype.eIsProxy()) {
			InternalEObject oldMtype = (InternalEObject)mtype;
			mtype = (mtype)eResolveProxy(oldMtype);
			if (mtype != oldMtype) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PromelaPackage.MTYPE_CONST__MTYPE, oldMtype, mtype));
			}
		}
		return mtype;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public mtype basicGetMtype() {
		return mtype;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMtype(mtype newMtype) {
		mtype oldMtype = mtype;
		mtype = newMtype;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.MTYPE_CONST__MTYPE, oldMtype, mtype));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.MTYPE_CONST__MTYPE:
				if (resolve) return getMtype();
				return basicGetMtype();
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
			case PromelaPackage.MTYPE_CONST__MTYPE:
				setMtype((mtype)newValue);
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
			case PromelaPackage.MTYPE_CONST__MTYPE:
				setMtype((mtype)null);
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
			case PromelaPackage.MTYPE_CONST__MTYPE:
				return mtype != null;
		}
		return super.eIsSet(featureID);
	}

} //mtype_constImpl
