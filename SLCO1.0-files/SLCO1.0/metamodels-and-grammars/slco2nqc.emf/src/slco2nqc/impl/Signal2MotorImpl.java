/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.impl;

import nqc.OutputModeEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import slco.Port;

import slco2nqc.Signal2Motor;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Signal2 Motor</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco2nqc.impl.Signal2MotorImpl#getSignalName <em>Signal Name</em>}</li>
 *   <li>{@link slco2nqc.impl.Signal2MotorImpl#getPort <em>Port</em>}</li>
 *   <li>{@link slco2nqc.impl.Signal2MotorImpl#getOutputMode <em>Output Mode</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class Signal2MotorImpl extends EObjectImpl implements Signal2Motor {
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
	 * The default value of the '{@link #getOutputMode() <em>Output Mode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOutputMode()
	 * @generated
	 * @ordered
	 */
	protected static final OutputModeEnum OUTPUT_MODE_EDEFAULT = OutputModeEnum.OUT_OFF;

	/**
	 * The cached value of the '{@link #getOutputMode() <em>Output Mode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOutputMode()
	 * @generated
	 * @ordered
	 */
	protected OutputModeEnum outputMode = OUTPUT_MODE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Signal2MotorImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return Slco2nqcPackage.Literals.SIGNAL2_MOTOR;
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
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SIGNAL2_MOTOR__SIGNAL_NAME, oldSignalName, signalName));
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
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, Slco2nqcPackage.SIGNAL2_MOTOR__PORT, oldPort, port));
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
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SIGNAL2_MOTOR__PORT, oldPort, port));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutputModeEnum getOutputMode() {
		return outputMode;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setOutputMode(OutputModeEnum newOutputMode) {
		OutputModeEnum oldOutputMode = outputMode;
		outputMode = newOutputMode == null ? OUTPUT_MODE_EDEFAULT : newOutputMode;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SIGNAL2_MOTOR__OUTPUT_MODE, oldOutputMode, outputMode));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case Slco2nqcPackage.SIGNAL2_MOTOR__SIGNAL_NAME:
				return getSignalName();
			case Slco2nqcPackage.SIGNAL2_MOTOR__PORT:
				if (resolve) return getPort();
				return basicGetPort();
			case Slco2nqcPackage.SIGNAL2_MOTOR__OUTPUT_MODE:
				return getOutputMode();
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
			case Slco2nqcPackage.SIGNAL2_MOTOR__SIGNAL_NAME:
				setSignalName((String)newValue);
				return;
			case Slco2nqcPackage.SIGNAL2_MOTOR__PORT:
				setPort((Port)newValue);
				return;
			case Slco2nqcPackage.SIGNAL2_MOTOR__OUTPUT_MODE:
				setOutputMode((OutputModeEnum)newValue);
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
			case Slco2nqcPackage.SIGNAL2_MOTOR__SIGNAL_NAME:
				setSignalName(SIGNAL_NAME_EDEFAULT);
				return;
			case Slco2nqcPackage.SIGNAL2_MOTOR__PORT:
				setPort((Port)null);
				return;
			case Slco2nqcPackage.SIGNAL2_MOTOR__OUTPUT_MODE:
				setOutputMode(OUTPUT_MODE_EDEFAULT);
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
			case Slco2nqcPackage.SIGNAL2_MOTOR__SIGNAL_NAME:
				return SIGNAL_NAME_EDEFAULT == null ? signalName != null : !SIGNAL_NAME_EDEFAULT.equals(signalName);
			case Slco2nqcPackage.SIGNAL2_MOTOR__PORT:
				return port != null;
			case Slco2nqcPackage.SIGNAL2_MOTOR__OUTPUT_MODE:
				return outputMode != OUTPUT_MODE_EDEFAULT;
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
		result.append(", outputMode: ");
		result.append(outputMode);
		result.append(')');
		return result.toString();
	}

} //Signal2MotorImpl
