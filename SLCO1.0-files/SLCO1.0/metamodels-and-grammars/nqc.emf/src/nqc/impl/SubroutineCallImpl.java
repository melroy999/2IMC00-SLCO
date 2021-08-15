/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.Subroutine;
import nqc.SubroutineCall;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Subroutine Call</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.SubroutineCallImpl#getCallee <em>Callee</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SubroutineCallImpl extends CallStatementImpl implements SubroutineCall {
	/**
	 * The cached value of the '{@link #getCallee() <em>Callee</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCallee()
	 * @generated
	 * @ordered
	 */
	protected Subroutine callee;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SubroutineCallImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getSubroutineCall();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Subroutine getCallee() {
		if (callee != null && callee.eIsProxy()) {
			InternalEObject oldCallee = (InternalEObject)callee;
			callee = (Subroutine)eResolveProxy(oldCallee);
			if (callee != oldCallee) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, NqcPackage.SUBROUTINE_CALL__CALLEE, oldCallee, callee));
			}
		}
		return callee;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Subroutine basicGetCallee() {
		return callee;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCallee(Subroutine newCallee) {
		Subroutine oldCallee = callee;
		callee = newCallee;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.SUBROUTINE_CALL__CALLEE, oldCallee, callee));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.SUBROUTINE_CALL__CALLEE:
				if (resolve) return getCallee();
				return basicGetCallee();
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
			case NqcPackage.SUBROUTINE_CALL__CALLEE:
				setCallee((Subroutine)newValue);
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
			case NqcPackage.SUBROUTINE_CALL__CALLEE:
				setCallee((Subroutine)null);
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
			case NqcPackage.SUBROUTINE_CALL__CALLEE:
				return callee != null;
		}
		return super.eIsSet(featureID);
	}

} //SubroutineCallImpl
