/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.NqcPackage;
import nqc.OutputPortNameConstant;
import nqc.OutputPortNameEnum;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Output Port Name Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.OutputPortNameConstantImpl#getOutputPortName <em>Output Port Name</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class OutputPortNameConstantImpl extends ConstantExpressionImpl implements OutputPortNameConstant {
	/**
	 * The default value of the '{@link #getOutputPortName() <em>Output Port Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOutputPortName()
	 * @generated
	 * @ordered
	 */
	protected static final OutputPortNameEnum OUTPUT_PORT_NAME_EDEFAULT = OutputPortNameEnum.OUT_A;

	/**
	 * The cached value of the '{@link #getOutputPortName() <em>Output Port Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOutputPortName()
	 * @generated
	 * @ordered
	 */
	protected OutputPortNameEnum outputPortName = OUTPUT_PORT_NAME_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected OutputPortNameConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getOutputPortNameConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutputPortNameEnum getOutputPortName() {
		return outputPortName;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setOutputPortName(OutputPortNameEnum newOutputPortName) {
		OutputPortNameEnum oldOutputPortName = outputPortName;
		outputPortName = newOutputPortName == null ? OUTPUT_PORT_NAME_EDEFAULT : newOutputPortName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.OUTPUT_PORT_NAME_CONSTANT__OUTPUT_PORT_NAME, oldOutputPortName, outputPortName));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.OUTPUT_PORT_NAME_CONSTANT__OUTPUT_PORT_NAME:
				return getOutputPortName();
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
			case NqcPackage.OUTPUT_PORT_NAME_CONSTANT__OUTPUT_PORT_NAME:
				setOutputPortName((OutputPortNameEnum)newValue);
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
			case NqcPackage.OUTPUT_PORT_NAME_CONSTANT__OUTPUT_PORT_NAME:
				setOutputPortName(OUTPUT_PORT_NAME_EDEFAULT);
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
			case NqcPackage.OUTPUT_PORT_NAME_CONSTANT__OUTPUT_PORT_NAME:
				return outputPortName != OUTPUT_PORT_NAME_EDEFAULT;
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
		result.append(" (OutputPortName: ");
		result.append(outputPortName);
		result.append(')');
		return result.toString();
	}

} //OutputPortNameConstantImpl
