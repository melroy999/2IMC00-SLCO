/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.BuiltInNullaryFunctionEnum;
import nqc.NqcPackage;
import nqc.NullaryBuiltInFunctionCall;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Nullary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.NullaryBuiltInFunctionCallImpl#getNullaryBuiltInFunction <em>Nullary Built In Function</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class NullaryBuiltInFunctionCallImpl extends BuiltInFunctionCallImpl implements NullaryBuiltInFunctionCall {
	/**
	 * The default value of the '{@link #getNullaryBuiltInFunction() <em>Nullary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getNullaryBuiltInFunction()
	 * @generated
	 * @ordered
	 */
	protected static final BuiltInNullaryFunctionEnum NULLARY_BUILT_IN_FUNCTION_EDEFAULT = BuiltInNullaryFunctionEnum.MUTE_SOUND;

	/**
	 * The cached value of the '{@link #getNullaryBuiltInFunction() <em>Nullary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getNullaryBuiltInFunction()
	 * @generated
	 * @ordered
	 */
	protected BuiltInNullaryFunctionEnum nullaryBuiltInFunction = NULLARY_BUILT_IN_FUNCTION_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected NullaryBuiltInFunctionCallImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getNullaryBuiltInFunctionCall();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInNullaryFunctionEnum getNullaryBuiltInFunction() {
		return nullaryBuiltInFunction;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setNullaryBuiltInFunction(BuiltInNullaryFunctionEnum newNullaryBuiltInFunction) {
		BuiltInNullaryFunctionEnum oldNullaryBuiltInFunction = nullaryBuiltInFunction;
		nullaryBuiltInFunction = newNullaryBuiltInFunction == null ? NULLARY_BUILT_IN_FUNCTION_EDEFAULT : newNullaryBuiltInFunction;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.NULLARY_BUILT_IN_FUNCTION_CALL__NULLARY_BUILT_IN_FUNCTION, oldNullaryBuiltInFunction, nullaryBuiltInFunction));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.NULLARY_BUILT_IN_FUNCTION_CALL__NULLARY_BUILT_IN_FUNCTION:
				return getNullaryBuiltInFunction();
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
			case NqcPackage.NULLARY_BUILT_IN_FUNCTION_CALL__NULLARY_BUILT_IN_FUNCTION:
				setNullaryBuiltInFunction((BuiltInNullaryFunctionEnum)newValue);
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
			case NqcPackage.NULLARY_BUILT_IN_FUNCTION_CALL__NULLARY_BUILT_IN_FUNCTION:
				setNullaryBuiltInFunction(NULLARY_BUILT_IN_FUNCTION_EDEFAULT);
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
			case NqcPackage.NULLARY_BUILT_IN_FUNCTION_CALL__NULLARY_BUILT_IN_FUNCTION:
				return nullaryBuiltInFunction != NULLARY_BUILT_IN_FUNCTION_EDEFAULT;
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
		result.append(" (NullaryBuiltInFunction: ");
		result.append(nullaryBuiltInFunction);
		result.append(')');
		return result.toString();
	}

} //NullaryBuiltInFunctionCallImpl
