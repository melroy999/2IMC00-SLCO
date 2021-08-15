/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.impl;

import nqc.SensorConfigEnum;
import nqc.SensorNameEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import slco.Port;

import slco2nqc.SensorInit;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Sensor Init</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco2nqc.impl.SensorInitImpl#getSensorName <em>Sensor Name</em>}</li>
 *   <li>{@link slco2nqc.impl.SensorInitImpl#getSensorConfiguration <em>Sensor Configuration</em>}</li>
 *   <li>{@link slco2nqc.impl.SensorInitImpl#getPort <em>Port</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SensorInitImpl extends EObjectImpl implements SensorInit {
	/**
	 * The default value of the '{@link #getSensorName() <em>Sensor Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSensorName()
	 * @generated
	 * @ordered
	 */
	protected static final SensorNameEnum SENSOR_NAME_EDEFAULT = SensorNameEnum.SENSOR_1;

	/**
	 * The cached value of the '{@link #getSensorName() <em>Sensor Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSensorName()
	 * @generated
	 * @ordered
	 */
	protected SensorNameEnum sensorName = SENSOR_NAME_EDEFAULT;

	/**
	 * The default value of the '{@link #getSensorConfiguration() <em>Sensor Configuration</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSensorConfiguration()
	 * @generated
	 * @ordered
	 */
	protected static final SensorConfigEnum SENSOR_CONFIGURATION_EDEFAULT = SensorConfigEnum.SENSOR_TOUCH;

	/**
	 * The cached value of the '{@link #getSensorConfiguration() <em>Sensor Configuration</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSensorConfiguration()
	 * @generated
	 * @ordered
	 */
	protected SensorConfigEnum sensorConfiguration = SENSOR_CONFIGURATION_EDEFAULT;

	/**
	 * The cached value of the '{@link #getPort() <em>Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPort()
	 * @generated
	 * @ordered
	 */
	protected Port port;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SensorInitImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return Slco2nqcPackage.Literals.SENSOR_INIT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorNameEnum getSensorName() {
		return sensorName;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSensorName(SensorNameEnum newSensorName) {
		SensorNameEnum oldSensorName = sensorName;
		sensorName = newSensorName == null ? SENSOR_NAME_EDEFAULT : newSensorName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SENSOR_INIT__SENSOR_NAME, oldSensorName, sensorName));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorConfigEnum getSensorConfiguration() {
		return sensorConfiguration;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSensorConfiguration(SensorConfigEnum newSensorConfiguration) {
		SensorConfigEnum oldSensorConfiguration = sensorConfiguration;
		sensorConfiguration = newSensorConfiguration == null ? SENSOR_CONFIGURATION_EDEFAULT : newSensorConfiguration;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SENSOR_INIT__SENSOR_CONFIGURATION, oldSensorConfiguration, sensorConfiguration));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port getPort() {
		if (port != null && port.eIsProxy()) {
			InternalEObject oldPort = (InternalEObject)port;
			port = (Port)eResolveProxy(oldPort);
			if (port != oldPort) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, Slco2nqcPackage.SENSOR_INIT__PORT, oldPort, port));
			}
		}
		return port;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port basicGetPort() {
		return port;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPort(Port newPort) {
		Port oldPort = port;
		port = newPort;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SENSOR_INIT__PORT, oldPort, port));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case Slco2nqcPackage.SENSOR_INIT__SENSOR_NAME:
				return getSensorName();
			case Slco2nqcPackage.SENSOR_INIT__SENSOR_CONFIGURATION:
				return getSensorConfiguration();
			case Slco2nqcPackage.SENSOR_INIT__PORT:
				if (resolve) return getPort();
				return basicGetPort();
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
			case Slco2nqcPackage.SENSOR_INIT__SENSOR_NAME:
				setSensorName((SensorNameEnum)newValue);
				return;
			case Slco2nqcPackage.SENSOR_INIT__SENSOR_CONFIGURATION:
				setSensorConfiguration((SensorConfigEnum)newValue);
				return;
			case Slco2nqcPackage.SENSOR_INIT__PORT:
				setPort((Port)newValue);
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
			case Slco2nqcPackage.SENSOR_INIT__SENSOR_NAME:
				setSensorName(SENSOR_NAME_EDEFAULT);
				return;
			case Slco2nqcPackage.SENSOR_INIT__SENSOR_CONFIGURATION:
				setSensorConfiguration(SENSOR_CONFIGURATION_EDEFAULT);
				return;
			case Slco2nqcPackage.SENSOR_INIT__PORT:
				setPort((Port)null);
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
			case Slco2nqcPackage.SENSOR_INIT__SENSOR_NAME:
				return sensorName != SENSOR_NAME_EDEFAULT;
			case Slco2nqcPackage.SENSOR_INIT__SENSOR_CONFIGURATION:
				return sensorConfiguration != SENSOR_CONFIGURATION_EDEFAULT;
			case Slco2nqcPackage.SENSOR_INIT__PORT:
				return port != null;
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
		result.append(" (sensorName: ");
		result.append(sensorName);
		result.append(", sensorConfiguration: ");
		result.append(sensorConfiguration);
		result.append(')');
		return result.toString();
	}

} //SensorInitImpl
