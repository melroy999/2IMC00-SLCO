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

import slco2nqc.SensorInit;
import slco2nqc.SensorInitList;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Sensor Init List</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco2nqc.impl.SensorInitListImpl#getSensorInits <em>Sensor Inits</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SensorInitListImpl extends EObjectImpl implements SensorInitList {
	/**
	 * The cached value of the '{@link #getSensorInits() <em>Sensor Inits</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSensorInits()
	 * @generated
	 * @ordered
	 */
	protected EList<SensorInit> sensorInits;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SensorInitListImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return Slco2nqcPackage.Literals.SENSOR_INIT_LIST;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<SensorInit> getSensorInits() {
		if (sensorInits == null) {
			sensorInits = new EObjectContainmentEList<SensorInit>(SensorInit.class, this, Slco2nqcPackage.SENSOR_INIT_LIST__SENSOR_INITS);
		}
		return sensorInits;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case Slco2nqcPackage.SENSOR_INIT_LIST__SENSOR_INITS:
				return ((InternalEList<?>)getSensorInits()).basicRemove(otherEnd, msgs);
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
			case Slco2nqcPackage.SENSOR_INIT_LIST__SENSOR_INITS:
				return getSensorInits();
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
			case Slco2nqcPackage.SENSOR_INIT_LIST__SENSOR_INITS:
				getSensorInits().clear();
				getSensorInits().addAll((Collection<? extends SensorInit>)newValue);
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
			case Slco2nqcPackage.SENSOR_INIT_LIST__SENSOR_INITS:
				getSensorInits().clear();
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
			case Slco2nqcPackage.SENSOR_INIT_LIST__SENSOR_INITS:
				return sensorInits != null && !sensorInits.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //SensorInitListImpl
