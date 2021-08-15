/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.BinaryBuiltInValueFunctionCall;
import nqc.BuiltInBinaryValueFunctionEnum;
import nqc.Expression;
import nqc.NqcPackage;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Binary Built In Value Function Call</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.BinaryBuiltInValueFunctionCallImpl#getBinaryBuiltInValueFunction <em>Binary Built In Value Function</em>}</li>
 *   <li>{@link nqc.impl.BinaryBuiltInValueFunctionCallImpl#getParameter1 <em>Parameter1</em>}</li>
 *   <li>{@link nqc.impl.BinaryBuiltInValueFunctionCallImpl#getParameter2 <em>Parameter2</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class BinaryBuiltInValueFunctionCallImpl extends BuiltInValueFunctionCallImpl implements BinaryBuiltInValueFunctionCall {
	/**
	 * The default value of the '{@link #getBinaryBuiltInValueFunction() <em>Binary Built In Value Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getBinaryBuiltInValueFunction()
	 * @generated
	 * @ordered
	 */
	protected static final BuiltInBinaryValueFunctionEnum BINARY_BUILT_IN_VALUE_FUNCTION_EDEFAULT = BuiltInBinaryValueFunctionEnum.SET_INDIRECT_VAR;

	/**
	 * The cached value of the '{@link #getBinaryBuiltInValueFunction() <em>Binary Built In Value Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getBinaryBuiltInValueFunction()
	 * @generated
	 * @ordered
	 */
	protected BuiltInBinaryValueFunctionEnum binaryBuiltInValueFunction = BINARY_BUILT_IN_VALUE_FUNCTION_EDEFAULT;

	/**
	 * The cached value of the '{@link #getParameter1() <em>Parameter1</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getParameter1()
	 * @generated
	 * @ordered
	 */
	protected Expression parameter1;

	/**
	 * The cached value of the '{@link #getParameter2() <em>Parameter2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getParameter2()
	 * @generated
	 * @ordered
	 */
	protected Expression parameter2;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected BinaryBuiltInValueFunctionCallImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getBinaryBuiltInValueFunctionCall();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInBinaryValueFunctionEnum getBinaryBuiltInValueFunction() {
		return binaryBuiltInValueFunction;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setBinaryBuiltInValueFunction(BuiltInBinaryValueFunctionEnum newBinaryBuiltInValueFunction) {
		BuiltInBinaryValueFunctionEnum oldBinaryBuiltInValueFunction = binaryBuiltInValueFunction;
		binaryBuiltInValueFunction = newBinaryBuiltInValueFunction == null ? BINARY_BUILT_IN_VALUE_FUNCTION_EDEFAULT : newBinaryBuiltInValueFunction;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__BINARY_BUILT_IN_VALUE_FUNCTION, oldBinaryBuiltInValueFunction, binaryBuiltInValueFunction));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getParameter1() {
		return parameter1;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetParameter1(Expression newParameter1, NotificationChain msgs) {
		Expression oldParameter1 = parameter1;
		parameter1 = newParameter1;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER1, oldParameter1, newParameter1);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setParameter1(Expression newParameter1) {
		if (newParameter1 != parameter1) {
			NotificationChain msgs = null;
			if (parameter1 != null)
				msgs = ((InternalEObject)parameter1).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER1, null, msgs);
			if (newParameter1 != null)
				msgs = ((InternalEObject)newParameter1).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER1, null, msgs);
			msgs = basicSetParameter1(newParameter1, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER1, newParameter1, newParameter1));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getParameter2() {
		return parameter2;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetParameter2(Expression newParameter2, NotificationChain msgs) {
		Expression oldParameter2 = parameter2;
		parameter2 = newParameter2;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER2, oldParameter2, newParameter2);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setParameter2(Expression newParameter2) {
		if (newParameter2 != parameter2) {
			NotificationChain msgs = null;
			if (parameter2 != null)
				msgs = ((InternalEObject)parameter2).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER2, null, msgs);
			if (newParameter2 != null)
				msgs = ((InternalEObject)newParameter2).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER2, null, msgs);
			msgs = basicSetParameter2(newParameter2, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER2, newParameter2, newParameter2));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER1:
				return basicSetParameter1(null, msgs);
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER2:
				return basicSetParameter2(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__BINARY_BUILT_IN_VALUE_FUNCTION:
				return getBinaryBuiltInValueFunction();
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER1:
				return getParameter1();
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER2:
				return getParameter2();
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
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__BINARY_BUILT_IN_VALUE_FUNCTION:
				setBinaryBuiltInValueFunction((BuiltInBinaryValueFunctionEnum)newValue);
				return;
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER1:
				setParameter1((Expression)newValue);
				return;
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER2:
				setParameter2((Expression)newValue);
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
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__BINARY_BUILT_IN_VALUE_FUNCTION:
				setBinaryBuiltInValueFunction(BINARY_BUILT_IN_VALUE_FUNCTION_EDEFAULT);
				return;
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER1:
				setParameter1((Expression)null);
				return;
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER2:
				setParameter2((Expression)null);
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
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__BINARY_BUILT_IN_VALUE_FUNCTION:
				return binaryBuiltInValueFunction != BINARY_BUILT_IN_VALUE_FUNCTION_EDEFAULT;
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER1:
				return parameter1 != null;
			case NqcPackage.BINARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER2:
				return parameter2 != null;
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
		result.append(" (BinaryBuiltInValueFunction: ");
		result.append(binaryBuiltInValueFunction);
		result.append(')');
		return result.toString();
	}

} //BinaryBuiltInValueFunctionCallImpl
