/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.BuiltInNullaryValueFunctionEnum;
import nqc.NqcPackage;
import nqc.NullaryBuiltInValueFunctionCall;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Nullary Built In Value Function Call</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.NullaryBuiltInValueFunctionCallImpl#getNullaryBuiltInValueFunction <em>Nullary Built In Value Function</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class NullaryBuiltInValueFunctionCallImpl extends BuiltInValueFunctionCallImpl implements NullaryBuiltInValueFunctionCall {
	/**
	 * The default value of the '{@link #getNullaryBuiltInValueFunction() <em>Nullary Built In Value Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getNullaryBuiltInValueFunction()
	 * @generated
	 * @ordered
	 */
	protected static final BuiltInNullaryValueFunctionEnum NULLARY_BUILT_IN_VALUE_FUNCTION_EDEFAULT = BuiltInNullaryValueFunctionEnum.CURRENT_EVENTS;

	/**
	 * The cached value of the '{@link #getNullaryBuiltInValueFunction() <em>Nullary Built In Value Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getNullaryBuiltInValueFunction()
	 * @generated
	 * @ordered
	 */
	protected BuiltInNullaryValueFunctionEnum nullaryBuiltInValueFunction = NULLARY_BUILT_IN_VALUE_FUNCTION_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected NullaryBuiltInValueFunctionCallImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getNullaryBuiltInValueFunctionCall();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInNullaryValueFunctionEnum getNullaryBuiltInValueFunction() {
		return nullaryBuiltInValueFunction;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setNullaryBuiltInValueFunction(BuiltInNullaryValueFunctionEnum newNullaryBuiltInValueFunction) {
		BuiltInNullaryValueFunctionEnum oldNullaryBuiltInValueFunction = nullaryBuiltInValueFunction;
		nullaryBuiltInValueFunction = newNullaryBuiltInValueFunction == null ? NULLARY_BUILT_IN_VALUE_FUNCTION_EDEFAULT : newNullaryBuiltInValueFunction;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.NULLARY_BUILT_IN_VALUE_FUNCTION_CALL__NULLARY_BUILT_IN_VALUE_FUNCTION, oldNullaryBuiltInValueFunction, nullaryBuiltInValueFunction));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.NULLARY_BUILT_IN_VALUE_FUNCTION_CALL__NULLARY_BUILT_IN_VALUE_FUNCTION:
				return getNullaryBuiltInValueFunction();
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
			case NqcPackage.NULLARY_BUILT_IN_VALUE_FUNCTION_CALL__NULLARY_BUILT_IN_VALUE_FUNCTION:
				setNullaryBuiltInValueFunction((BuiltInNullaryValueFunctionEnum)newValue);
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
			case NqcPackage.NULLARY_BUILT_IN_VALUE_FUNCTION_CALL__NULLARY_BUILT_IN_VALUE_FUNCTION:
				setNullaryBuiltInValueFunction(NULLARY_BUILT_IN_VALUE_FUNCTION_EDEFAULT);
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
			case NqcPackage.NULLARY_BUILT_IN_VALUE_FUNCTION_CALL__NULLARY_BUILT_IN_VALUE_FUNCTION:
				return nullaryBuiltInValueFunction != NULLARY_BUILT_IN_VALUE_FUNCTION_EDEFAULT;
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
		result.append(" (NullaryBuiltInValueFunction: ");
		result.append(nullaryBuiltInValueFunction);
		result.append(')');
		return result.toString();
	}

} //NullaryBuiltInValueFunctionCallImpl
