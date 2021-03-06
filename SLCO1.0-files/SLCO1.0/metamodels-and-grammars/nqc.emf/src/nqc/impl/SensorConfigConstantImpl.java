/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.SensorConfigConstant;
import nqc.SensorConfigEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Sensor Config Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.SensorConfigConstantImpl#getSensorConfig <em>Sensor Config</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SensorConfigConstantImpl extends ConstantExpressionImpl implements SensorConfigConstant {
	/**
	 * The default value of the '{@link #getSensorConfig() <em>Sensor Config</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSensorConfig()
	 * @generated
	 * @ordered
	 */
	protected static final SensorConfigEnum SENSOR_CONFIG_EDEFAULT = SensorConfigEnum.SENSOR_TOUCH;

	/**
	 * The cached value of the '{@link #getSensorConfig() <em>Sensor Config</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSensorConfig()
	 * @generated
	 * @ordered
	 */
	protected SensorConfigEnum sensorConfig = SENSOR_CONFIG_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SensorConfigConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getSensorConfigConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorConfigEnum getSensorConfig() {
		return sensorConfig;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSensorConfig(SensorConfigEnum newSensorConfig) {
		SensorConfigEnum oldSensorConfig = sensorConfig;
		sensorConfig = newSensorConfig == null ? SENSOR_CONFIG_EDEFAULT : newSensorConfig;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.SENSOR_CONFIG_CONSTANT__SENSOR_CONFIG, oldSensorConfig, sensorConfig));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.SENSOR_CONFIG_CONSTANT__SENSOR_CONFIG:
				return getSensorConfig();
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
			case NqcPackage.SENSOR_CONFIG_CONSTANT__SENSOR_CONFIG:
				setSensorConfig((SensorConfigEnum)newValue);
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
			case NqcPackage.SENSOR_CONFIG_CONSTANT__SENSOR_CONFIG:
				setSensorConfig(SENSOR_CONFIG_EDEFAULT);
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
			case NqcPackage.SENSOR_CONFIG_CONSTANT__SENSOR_CONFIG:
				return sensorConfig != SENSOR_CONFIG_EDEFAULT;
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
		result.append(" (SensorConfig: ");
		result.append(sensorConfig);
		result.append(')');
		return result.toString();
	}

} //SensorConfigConstantImpl
