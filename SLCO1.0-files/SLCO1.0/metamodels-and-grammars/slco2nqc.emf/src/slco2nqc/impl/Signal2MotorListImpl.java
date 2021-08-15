/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import slco2nqc.Signal2Motor;
import slco2nqc.Signal2MotorList;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Signal2 Motor List</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco2nqc.impl.Signal2MotorListImpl#getSignal2Motors <em>Signal2 Motors</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class Signal2MotorListImpl extends EObjectImpl implements Signal2MotorList {
	/**
	 * The cached value of the '{@link #getSignal2Motors() <em>Signal2 Motors</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSignal2Motors()
	 * @generated
	 * @ordered
	 */
	protected EList<Signal2Motor> signal2Motors;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Signal2MotorListImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return Slco2nqcPackage.Literals.SIGNAL2_MOTOR_LIST;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Signal2Motor> getSignal2Motors() {
		if (signal2Motors == null) {
			signal2Motors = new EObjectContainmentEList<Signal2Motor>(Signal2Motor.class, this, Slco2nqcPackage.SIGNAL2_MOTOR_LIST__SIGNAL2_MOTORS);
		}
		return signal2Motors;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case Slco2nqcPackage.SIGNAL2_MOTOR_LIST__SIGNAL2_MOTORS:
				return ((InternalEList<?>)getSignal2Motors()).basicRemove(otherEnd, msgs);
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
			case Slco2nqcPackage.SIGNAL2_MOTOR_LIST__SIGNAL2_MOTORS:
				return getSignal2Motors();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case Slco2nqcPackage.SIGNAL2_MOTOR_LIST__SIGNAL2_MOTORS:
				getSignal2Motors().clear();
				getSignal2Motors().addAll((Collection<? extends Signal2Motor>)newValue);
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
			case Slco2nqcPackage.SIGNAL2_MOTOR_LIST__SIGNAL2_MOTORS:
				getSignal2Motors().clear();
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
			case Slco2nqcPackage.SIGNAL2_MOTOR_LIST__SIGNAL2_MOTORS:
				return signal2Motors != null && !signal2Motors.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //Signal2MotorListImpl
