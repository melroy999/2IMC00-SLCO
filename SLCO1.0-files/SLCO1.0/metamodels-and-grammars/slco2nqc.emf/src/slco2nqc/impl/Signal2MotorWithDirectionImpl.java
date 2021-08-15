/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.impl;

import nqc.DirectionEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import slco2nqc.Signal2MotorWithDirection;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Signal2 Motor With Direction</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco2nqc.impl.Signal2MotorWithDirectionImpl#getOutputDirection <em>Output Direction</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class Signal2MotorWithDirectionImpl extends Signal2MotorImpl implements Signal2MotorWithDirection {
	/**
	 * The default value of the '{@link #getOutputDirection() <em>Output Direction</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOutputDirection()
	 * @generated
	 * @ordered
	 */
	protected static final DirectionEnum OUTPUT_DIRECTION_EDEFAULT = DirectionEnum.OUT_FWD;

	/**
	 * The cached value of the '{@link #getOutputDirection() <em>Output Direction</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOutputDirection()
	 * @generated
	 * @ordered
	 */
	protected DirectionEnum outputDirection = OUTPUT_DIRECTION_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Signal2MotorWithDirectionImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return Slco2nqcPackage.Literals.SIGNAL2_MOTOR_WITH_DIRECTION;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DirectionEnum getOutputDirection() {
		return outputDirection;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setOutputDirection(DirectionEnum newOutputDirection) {
		DirectionEnum oldOutputDirection = outputDirection;
		outputDirection = newOutputDirection == null ? OUTPUT_DIRECTION_EDEFAULT : newOutputDirection;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SIGNAL2_MOTOR_WITH_DIRECTION__OUTPUT_DIRECTION, oldOutputDirection, outputDirection));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case Slco2nqcPackage.SIGNAL2_MOTOR_WITH_DIRECTION__OUTPUT_DIRECTION:
				return getOutputDirection();
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
			case Slco2nqcPackage.SIGNAL2_MOTOR_WITH_DIRECTION__OUTPUT_DIRECTION:
				setOutputDirection((DirectionEnum)newValue);
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
			case Slco2nqcPackage.SIGNAL2_MOTOR_WITH_DIRECTION__OUTPUT_DIRECTION:
				setOutputDirection(OUTPUT_DIRECTION_EDEFAULT);
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
			case Slco2nqcPackage.SIGNAL2_MOTOR_WITH_DIRECTION__OUTPUT_DIRECTION:
				return outputDirection != OUTPUT_DIRECTION_EDEFAULT;
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
		result.append(" (outputDirection: ");
		result.append(outputDirection);
		result.append(')');
		return result.toString();
	}

} //Signal2MotorWithDirectionImpl
