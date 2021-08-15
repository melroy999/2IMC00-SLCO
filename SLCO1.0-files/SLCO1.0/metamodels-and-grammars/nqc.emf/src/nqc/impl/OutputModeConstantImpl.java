/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.OutputModeConstant;
import nqc.OutputModeEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Output Mode Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.OutputModeConstantImpl#getOutputMode <em>Output Mode</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class OutputModeConstantImpl extends ConstantExpressionImpl implements OutputModeConstant {
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
	protected OutputModeConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getOutputModeConstant();
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
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.OUTPUT_MODE_CONSTANT__OUTPUT_MODE, oldOutputMode, outputMode));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.OUTPUT_MODE_CONSTANT__OUTPUT_MODE:
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
			case NqcPackage.OUTPUT_MODE_CONSTANT__OUTPUT_MODE:
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
			case NqcPackage.OUTPUT_MODE_CONSTANT__OUTPUT_MODE:
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
			case NqcPackage.OUTPUT_MODE_CONSTANT__OUTPUT_MODE:
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
		result.append(" (OutputMode: ");
		result.append(outputMode);
		result.append(')');
		return result.toString();
	}

} //OutputModeConstantImpl
