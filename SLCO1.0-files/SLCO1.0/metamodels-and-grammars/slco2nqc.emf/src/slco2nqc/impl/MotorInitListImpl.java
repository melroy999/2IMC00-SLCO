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

import slco2nqc.MotorInit;
import slco2nqc.MotorInitList;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Motor Init List</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco2nqc.impl.MotorInitListImpl#getMotorInits <em>Motor Inits</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class MotorInitListImpl extends EObjectImpl implements MotorInitList {
	/**
	 * The cached value of the '{@link #getMotorInits() <em>Motor Inits</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMotorInits()
	 * @generated
	 * @ordered
	 */
	protected EList<MotorInit> motorInits;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MotorInitListImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return Slco2nqcPackage.Literals.MOTOR_INIT_LIST;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<MotorInit> getMotorInits() {
		if (motorInits == null) {
			motorInits = new EObjectContainmentEList<MotorInit>(MotorInit.class, this, Slco2nqcPackage.MOTOR_INIT_LIST__MOTOR_INITS);
		}
		return motorInits;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case Slco2nqcPackage.MOTOR_INIT_LIST__MOTOR_INITS:
				return ((InternalEList<?>)getMotorInits()).basicRemove(otherEnd, msgs);
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
			case Slco2nqcPackage.MOTOR_INIT_LIST__MOTOR_INITS:
				return getMotorInits();
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
			case Slco2nqcPackage.MOTOR_INIT_LIST__MOTOR_INITS:
				getMotorInits().clear();
				getMotorInits().addAll((Collection<? extends MotorInit>)newValue);
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
			case Slco2nqcPackage.MOTOR_INIT_LIST__MOTOR_INITS:
				getMotorInits().clear();
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
			case Slco2nqcPackage.MOTOR_INIT_LIST__MOTOR_INITS:
				return motorInits != null && !motorInits.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //MotorInitListImpl
