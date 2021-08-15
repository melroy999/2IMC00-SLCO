/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import promela.PromelaPackage;
import promela.recv_arg_varref;
import promela.varref;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>recv arg varref</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.recv_arg_varrefImpl#getVarref <em>Varref</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class recv_arg_varrefImpl extends recv_argImpl implements recv_arg_varref {
	/**
	 * The cached value of the '{@link #getVarref() <em>Varref</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVarref()
	 * @generated
	 * @ordered
	 */
	protected varref varref;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected recv_arg_varrefImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.RECV_ARG_VARREF;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public varref getVarref() {
		return varref;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetVarref(varref newVarref, NotificationChain msgs) {
		varref oldVarref = varref;
		varref = newVarref;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.RECV_ARG_VARREF__VARREF, oldVarref, newVarref);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setVarref(varref newVarref) {
		if (newVarref != varref) {
			NotificationChain msgs = null;
			if (varref != null)
				msgs = ((InternalEObject)varref).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.RECV_ARG_VARREF__VARREF, null, msgs);
			if (newVarref != null)
				msgs = ((InternalEObject)newVarref).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.RECV_ARG_VARREF__VARREF, null, msgs);
			msgs = basicSetVarref(newVarref, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.RECV_ARG_VARREF__VARREF, newVarref, newVarref));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.RECV_ARG_VARREF__VARREF:
				return basicSetVarref(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.RECV_ARG_VARREF__VARREF:
				return getVarref();
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
			case PromelaPackage.RECV_ARG_VARREF__VARREF:
				setVarref((varref)newValue);
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
			case PromelaPackage.RECV_ARG_VARREF__VARREF:
				setVarref((varref)null);
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
			case PromelaPackage.RECV_ARG_VARREF__VARREF:
				return varref != null;
		}
		return super.eIsSet(featureID);
	}

} //recv_arg_varrefImpl
