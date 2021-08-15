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
import promela.goto_stmnt;
import promela.stmnt;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>goto stmnt</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.goto_stmntImpl#getStmnt <em>Stmnt</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class goto_stmntImpl extends stmntImpl implements goto_stmnt {
	/**
	 * The cached value of the '{@link #getStmnt() <em>Stmnt</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getStmnt()
	 * @generated
	 * @ordered
	 */
	protected stmnt stmnt;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected goto_stmntImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.GOTO_STMNT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public stmnt getStmnt() {
		if (stmnt != null && stmnt.eIsProxy()) {
			InternalEObject oldStmnt = (InternalEObject)stmnt;
			stmnt = (stmnt)eResolveProxy(oldStmnt);
			if (stmnt != oldStmnt) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PromelaPackage.GOTO_STMNT__STMNT, oldStmnt, stmnt));
			}
		}
		return stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public stmnt basicGetStmnt() {
		return stmnt;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setStmnt(stmnt newStmnt) {
		stmnt oldStmnt = stmnt;
		stmnt = newStmnt;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.GOTO_STMNT__STMNT, oldStmnt, stmnt));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.GOTO_STMNT__STMNT:
				if (resolve) return getStmnt();
				return basicGetStmnt();
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
			case PromelaPackage.GOTO_STMNT__STMNT:
				setStmnt((stmnt)newValue);
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
			case PromelaPackage.GOTO_STMNT__STMNT:
				setStmnt((stmnt)null);
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
			case PromelaPackage.GOTO_STMNT__STMNT:
				return stmnt != null;
		}
		return super.eIsSet(featureID);
	}

} //goto_stmntImpl
