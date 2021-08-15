/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.SensorModeConstant;
import nqc.SensorModeEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Sensor Mode Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.SensorModeConstantImpl#getSensorMode <em>Sensor Mode</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SensorModeConstantImpl extends ConstantExpressionImpl implements SensorModeConstant {
	/**
	 * The default value of the '{@link #getSensorMode() <em>Sensor Mode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSensorMode()
	 * @generated
	 * @ordered
	 */
	protected static final SensorModeEnum SENSOR_MODE_EDEFAULT = SensorModeEnum.SENSOR_MODE_RAW;

	/**
	 * The cached value of the '{@link #getSensorMode() <em>Sensor Mode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSensorMode()
	 * @generated
	 * @ordered
	 */
	protected SensorModeEnum sensorMode = SENSOR_MODE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SensorModeConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getSensorModeConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorModeEnum getSensorMode() {
		return sensorMode;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSensorMode(SensorModeEnum newSensorMode) {
		SensorModeEnum oldSensorMode = sensorMode;
		sensorMode = newSensorMode == null ? SENSOR_MODE_EDEFAULT : newSensorMode;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.SENSOR_MODE_CONSTANT__SENSOR_MODE, oldSensorMode, sensorMode));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.SENSOR_MODE_CONSTANT__SENSOR_MODE:
				return getSensorMode();
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
			case NqcPackage.SENSOR_MODE_CONSTANT__SENSOR_MODE:
				setSensorMode((SensorModeEnum)newValue);
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
			case NqcPackage.SENSOR_MODE_CONSTANT__SENSOR_MODE:
				setSensorMode(SENSOR_MODE_EDEFAULT);
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
			case NqcPackage.SENSOR_MODE_CONSTANT__SENSOR_MODE:
				return sensorMode != SENSOR_MODE_EDEFAULT;
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
		result.append(" (SensorMode: ");
		result.append(sensorMode);
		result.append(')');
		return result.toString();
	}

} //SensorModeConstantImpl
