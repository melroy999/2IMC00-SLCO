/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.impl;

import nqc.OutputPortNameEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import slco.Port;

import slco2nqc.MotorInit;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Motor Init</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco2nqc.impl.MotorInitImpl#getPort <em>Port</em>}</li>
 *   <li>{@link slco2nqc.impl.MotorInitImpl#getPower <em>Power</em>}</li>
 *   <li>{@link slco2nqc.impl.MotorInitImpl#getOutputName <em>Output Name</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class MotorInitImpl extends EObjectImpl implements MotorInit {
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
	 * The default value of the '{@link #getPower() <em>Power</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPower()
	 * @generated
	 * @ordered
	 */
	protected static final int POWER_EDEFAULT = 0;

	/**
	 * The cached value of the '{@link #getPower() <em>Power</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPower()
	 * @generated
	 * @ordered
	 */
	protected int power = POWER_EDEFAULT;

	/**
	 * The default value of the '{@link #getOutputName() <em>Output Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOutputName()
	 * @generated
	 * @ordered
	 */
	protected static final OutputPortNameEnum OUTPUT_NAME_EDEFAULT = OutputPortNameEnum.OUT_A;

	/**
	 * The cached value of the '{@link #getOutputName() <em>Output Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOutputName()
	 * @generated
	 * @ordered
	 */
	protected OutputPortNameEnum outputName = OUTPUT_NAME_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MotorInitImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return Slco2nqcPackage.Literals.MOTOR_INIT;
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
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, Slco2nqcPackage.MOTOR_INIT__PORT, oldPort, port));
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
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.MOTOR_INIT__PORT, oldPort, port));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getPower() {
		return power;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPower(int newPower) {
		int oldPower = power;
		power = newPower;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.MOTOR_INIT__POWER, oldPower, power));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutputPortNameEnum getOutputName() {
		return outputName;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setOutputName(OutputPortNameEnum newOutputName) {
		OutputPortNameEnum oldOutputName = outputName;
		outputName = newOutputName == null ? OUTPUT_NAME_EDEFAULT : newOutputName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.MOTOR_INIT__OUTPUT_NAME, oldOutputName, outputName));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case Slco2nqcPackage.MOTOR_INIT__PORT:
				if (resolve) return getPort();
				return basicGetPort();
			case Slco2nqcPackage.MOTOR_INIT__POWER:
				return getPower();
			case Slco2nqcPackage.MOTOR_INIT__OUTPUT_NAME:
				return getOutputName();
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
			case Slco2nqcPackage.MOTOR_INIT__PORT:
				setPort((Port)newValue);
				return;
			case Slco2nqcPackage.MOTOR_INIT__POWER:
				setPower((Integer)newValue);
				return;
			case Slco2nqcPackage.MOTOR_INIT__OUTPUT_NAME:
				setOutputName((OutputPortNameEnum)newValue);
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
			case Slco2nqcPackage.MOTOR_INIT__PORT:
				setPort((Port)null);
				return;
			case Slco2nqcPackage.MOTOR_INIT__POWER:
				setPower(POWER_EDEFAULT);
				return;
			case Slco2nqcPackage.MOTOR_INIT__OUTPUT_NAME:
				setOutputName(OUTPUT_NAME_EDEFAULT);
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
			case Slco2nqcPackage.MOTOR_INIT__PORT:
				return port != null;
			case Slco2nqcPackage.MOTOR_INIT__POWER:
				return power != POWER_EDEFAULT;
			case Slco2nqcPackage.MOTOR_INIT__OUTPUT_NAME:
				return outputName != OUTPUT_NAME_EDEFAULT;
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
		result.append(" (power: ");
		result.append(power);
		result.append(", outputName: ");
		result.append(outputName);
		result.append(')');
		return result.toString();
	}

} //MotorInitImpl
