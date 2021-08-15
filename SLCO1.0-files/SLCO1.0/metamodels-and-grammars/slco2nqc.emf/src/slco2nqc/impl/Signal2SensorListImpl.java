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

import slco2nqc.Signal2Sensor;
import slco2nqc.Signal2SensorList;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Signal2 Sensor List</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco2nqc.impl.Signal2SensorListImpl#getSignal2Sensors <em>Signal2 Sensors</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class Signal2SensorListImpl extends EObjectImpl implements Signal2SensorList {
	/**
	 * The cached value of the '{@link #getSignal2Sensors() <em>Signal2 Sensors</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSignal2Sensors()
	 * @generated
	 * @ordered
	 */
	protected EList<Signal2Sensor> signal2Sensors;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Signal2SensorListImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return Slco2nqcPackage.Literals.SIGNAL2_SENSOR_LIST;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Signal2Sensor> getSignal2Sensors() {
		if (signal2Sensors == null) {
			signal2Sensors = new EObjectContainmentEList<Signal2Sensor>(Signal2Sensor.class, this, Slco2nqcPackage.SIGNAL2_SENSOR_LIST__SIGNAL2_SENSORS);
		}
		return signal2Sensors;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case Slco2nqcPackage.SIGNAL2_SENSOR_LIST__SIGNAL2_SENSORS:
				return ((InternalEList<?>)getSignal2Sensors()).basicRemove(otherEnd, msgs);
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
			case Slco2nqcPackage.SIGNAL2_SENSOR_LIST__SIGNAL2_SENSORS:
				return getSignal2Sensors();
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
			case Slco2nqcPackage.SIGNAL2_SENSOR_LIST__SIGNAL2_SENSORS:
				getSignal2Sensors().clear();
				getSignal2Sensors().addAll((Collection<? extends Signal2Sensor>)newValue);
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
			case Slco2nqcPackage.SIGNAL2_SENSOR_LIST__SIGNAL2_SENSORS:
				getSignal2Sensors().clear();
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
			case Slco2nqcPackage.SIGNAL2_SENSOR_LIST__SIGNAL2_SENSORS:
				return signal2Sensors != null && !signal2Sensors.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //Signal2SensorListImpl
