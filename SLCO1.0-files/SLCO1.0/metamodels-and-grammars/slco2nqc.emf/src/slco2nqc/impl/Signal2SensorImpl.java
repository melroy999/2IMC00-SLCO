/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.impl;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import slco.Port;

import slco2nqc.Signal2Sensor;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Signal2 Sensor</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco2nqc.impl.Signal2SensorImpl#getSignalName <em>Signal Name</em>}</li>
 *   <li>{@link slco2nqc.impl.Signal2SensorImpl#getPort <em>Port</em>}</li>
 *   <li>{@link slco2nqc.impl.Signal2SensorImpl#isSensorIsHigh <em>Sensor Is High</em>}</li>
 *   <li>{@link slco2nqc.impl.Signal2SensorImpl#getTreshold <em>Treshold</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class Signal2SensorImpl extends EObjectImpl implements Signal2Sensor {
	/**
	 * The default value of the '{@link #getSignalName() <em>Signal Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSignalName()
	 * @generated
	 * @ordered
	 */
	protected static final String SIGNAL_NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getSignalName() <em>Signal Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSignalName()
	 * @generated
	 * @ordered
	 */
	protected String signalName = SIGNAL_NAME_EDEFAULT;

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
	 * The default value of the '{@link #isSensorIsHigh() <em>Sensor Is High</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isSensorIsHigh()
	 * @generated
	 * @ordered
	 */
	protected static final boolean SENSOR_IS_HIGH_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isSensorIsHigh() <em>Sensor Is High</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isSensorIsHigh()
	 * @generated
	 * @ordered
	 */
	protected boolean sensorIsHigh = SENSOR_IS_HIGH_EDEFAULT;

	/**
	 * The default value of the '{@link #getTreshold() <em>Treshold</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTreshold()
	 * @generated
	 * @ordered
	 */
	protected static final int TRESHOLD_EDEFAULT = 0;

	/**
	 * The cached value of the '{@link #getTreshold() <em>Treshold</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTreshold()
	 * @generated
	 * @ordered
	 */
	protected int treshold = TRESHOLD_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Signal2SensorImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return Slco2nqcPackage.Literals.SIGNAL2_SENSOR;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getSignalName() {
		return signalName;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSignalName(String newSignalName) {
		String oldSignalName = signalName;
		signalName = newSignalName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SIGNAL2_SENSOR__SIGNAL_NAME, oldSignalName, signalName));
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
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, Slco2nqcPackage.SIGNAL2_SENSOR__PORT, oldPort, port));
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
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SIGNAL2_SENSOR__PORT, oldPort, port));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public boolean isSensorIsHigh() {
		return sensorIsHigh;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSensorIsHigh(boolean newSensorIsHigh) {
		boolean oldSensorIsHigh = sensorIsHigh;
		sensorIsHigh = newSensorIsHigh;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SIGNAL2_SENSOR__SENSOR_IS_HIGH, oldSensorIsHigh, sensorIsHigh));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getTreshold() {
		return treshold;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setTreshold(int newTreshold) {
		int oldTreshold = treshold;
		treshold = newTreshold;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SIGNAL2_SENSOR__TRESHOLD, oldTreshold, treshold));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case Slco2nqcPackage.SIGNAL2_SENSOR__SIGNAL_NAME:
				return getSignalName();
			case Slco2nqcPackage.SIGNAL2_SENSOR__PORT:
				if (resolve) return getPort();
				return basicGetPort();
			case Slco2nqcPackage.SIGNAL2_SENSOR__SENSOR_IS_HIGH:
				return isSensorIsHigh();
			case Slco2nqcPackage.SIGNAL2_SENSOR__TRESHOLD:
				return getTreshold();
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
			case Slco2nqcPackage.SIGNAL2_SENSOR__SIGNAL_NAME:
				setSignalName((String)newValue);
				return;
			case Slco2nqcPackage.SIGNAL2_SENSOR__PORT:
				setPort((Port)newValue);
				return;
			case Slco2nqcPackage.SIGNAL2_SENSOR__SENSOR_IS_HIGH:
				setSensorIsHigh((Boolean)newValue);
				return;
			case Slco2nqcPackage.SIGNAL2_SENSOR__TRESHOLD:
				setTreshold((Integer)newValue);
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
			case Slco2nqcPackage.SIGNAL2_SENSOR__SIGNAL_NAME:
				setSignalName(SIGNAL_NAME_EDEFAULT);
				return;
			case Slco2nqcPackage.SIGNAL2_SENSOR__PORT:
				setPort((Port)null);
				return;
			case Slco2nqcPackage.SIGNAL2_SENSOR__SENSOR_IS_HIGH:
				setSensorIsHigh(SENSOR_IS_HIGH_EDEFAULT);
				return;
			case Slco2nqcPackage.SIGNAL2_SENSOR__TRESHOLD:
				setTreshold(TRESHOLD_EDEFAULT);
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
			case Slco2nqcPackage.SIGNAL2_SENSOR__SIGNAL_NAME:
				return SIGNAL_NAME_EDEFAULT == null ? signalName != null : !SIGNAL_NAME_EDEFAULT.equals(signalName);
			case Slco2nqcPackage.SIGNAL2_SENSOR__PORT:
				return port != null;
			case Slco2nqcPackage.SIGNAL2_SENSOR__SENSOR_IS_HIGH:
				return sensorIsHigh != SENSOR_IS_HIGH_EDEFAULT;
			case Slco2nqcPackage.SIGNAL2_SENSOR__TRESHOLD:
				return treshold != TRESHOLD_EDEFAULT;
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
		result.append(" (signalName: ");
		result.append(signalName);
		result.append(", sensorIsHigh: ");
		result.append(sensorIsHigh);
		result.append(", treshold: ");
		result.append(treshold);
		result.append(')');
		return result.toString();
	}

} //Signal2SensorImpl
