/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.SensorNameConstant;
import nqc.SensorNameEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Sensor Name Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.SensorNameConstantImpl#getSensorName <em>Sensor Name</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SensorNameConstantImpl extends ConstantExpressionImpl implements SensorNameConstant {
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
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SensorNameConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getSensorNameConstant();
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
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.SENSOR_NAME_CONSTANT__SENSOR_NAME, oldSensorName, sensorName));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.SENSOR_NAME_CONSTANT__SENSOR_NAME:
				return getSensorName();
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
			case NqcPackage.SENSOR_NAME_CONSTANT__SENSOR_NAME:
				setSensorName((SensorNameEnum)newValue);
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
			case NqcPackage.SENSOR_NAME_CONSTANT__SENSOR_NAME:
				setSensorName(SENSOR_NAME_EDEFAULT);
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
			case NqcPackage.SENSOR_NAME_CONSTANT__SENSOR_NAME:
				return sensorName != SENSOR_NAME_EDEFAULT;
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
		result.append(" (SensorName: ");
		result.append(sensorName);
		result.append(')');
		return result.toString();
	}

} //SensorNameConstantImpl
