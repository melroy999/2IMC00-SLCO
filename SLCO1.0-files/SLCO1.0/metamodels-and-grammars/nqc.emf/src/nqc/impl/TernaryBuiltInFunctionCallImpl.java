/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.BuiltInTernaryFunctionEnum;
import nqc.Expression;
import nqc.NqcPackage;
import nqc.TernaryBuiltInFunctionCall;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Ternary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.TernaryBuiltInFunctionCallImpl#getTernaryBuiltInFunction <em>Ternary Built In Function</em>}</li>
 *   <li>{@link nqc.impl.TernaryBuiltInFunctionCallImpl#getParameter1 <em>Parameter1</em>}</li>
 *   <li>{@link nqc.impl.TernaryBuiltInFunctionCallImpl#getParameter2 <em>Parameter2</em>}</li>
 *   <li>{@link nqc.impl.TernaryBuiltInFunctionCallImpl#getParameter3 <em>Parameter3</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class TernaryBuiltInFunctionCallImpl extends BuiltInFunctionCallImpl implements TernaryBuiltInFunctionCall {
	/**
	 * The default value of the '{@link #getTernaryBuiltInFunction() <em>Ternary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTernaryBuiltInFunction()
	 * @generated
	 * @ordered
	 */
	protected static final BuiltInTernaryFunctionEnum TERNARY_BUILT_IN_FUNCTION_EDEFAULT = BuiltInTernaryFunctionEnum.SET_SPYBOT_PING;

	/**
	 * The cached value of the '{@link #getTernaryBuiltInFunction() <em>Ternary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTernaryBuiltInFunction()
	 * @generated
	 * @ordered
	 */
	protected BuiltInTernaryFunctionEnum ternaryBuiltInFunction = TERNARY_BUILT_IN_FUNCTION_EDEFAULT;

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
	 * The cached value of the '{@link #getParameter3() <em>Parameter3</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getParameter3()
	 * @generated
	 * @ordered
	 */
	protected Expression parameter3;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected TernaryBuiltInFunctionCallImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getTernaryBuiltInFunctionCall();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInTernaryFunctionEnum getTernaryBuiltInFunction() {
		return ternaryBuiltInFunction;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setTernaryBuiltInFunction(BuiltInTernaryFunctionEnum newTernaryBuiltInFunction) {
		BuiltInTernaryFunctionEnum oldTernaryBuiltInFunction = ternaryBuiltInFunction;
		ternaryBuiltInFunction = newTernaryBuiltInFunction == null ? TERNARY_BUILT_IN_FUNCTION_EDEFAULT : newTernaryBuiltInFunction;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__TERNARY_BUILT_IN_FUNCTION, oldTernaryBuiltInFunction, ternaryBuiltInFunction));
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1, oldParameter1, newParameter1);
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
				msgs = ((InternalEObject)parameter1).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1, null, msgs);
			if (newParameter1 != null)
				msgs = ((InternalEObject)newParameter1).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1, null, msgs);
			msgs = basicSetParameter1(newParameter1, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1, newParameter1, newParameter1));
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2, oldParameter2, newParameter2);
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
				msgs = ((InternalEObject)parameter2).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2, null, msgs);
			if (newParameter2 != null)
				msgs = ((InternalEObject)newParameter2).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2, null, msgs);
			msgs = basicSetParameter2(newParameter2, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2, newParameter2, newParameter2));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getParameter3() {
		return parameter3;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetParameter3(Expression newParameter3, NotificationChain msgs) {
		Expression oldParameter3 = parameter3;
		parameter3 = newParameter3;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3, oldParameter3, newParameter3);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setParameter3(Expression newParameter3) {
		if (newParameter3 != parameter3) {
			NotificationChain msgs = null;
			if (parameter3 != null)
				msgs = ((InternalEObject)parameter3).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3, null, msgs);
			if (newParameter3 != null)
				msgs = ((InternalEObject)newParameter3).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3, null, msgs);
			msgs = basicSetParameter3(newParameter3, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3, newParameter3, newParameter3));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1:
				return basicSetParameter1(null, msgs);
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2:
				return basicSetParameter2(null, msgs);
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3:
				return basicSetParameter3(null, msgs);
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
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__TERNARY_BUILT_IN_FUNCTION:
				return getTernaryBuiltInFunction();
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1:
				return getParameter1();
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2:
				return getParameter2();
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3:
				return getParameter3();
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
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__TERNARY_BUILT_IN_FUNCTION:
				setTernaryBuiltInFunction((BuiltInTernaryFunctionEnum)newValue);
				return;
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1:
				setParameter1((Expression)newValue);
				return;
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2:
				setParameter2((Expression)newValue);
				return;
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3:
				setParameter3((Expression)newValue);
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
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__TERNARY_BUILT_IN_FUNCTION:
				setTernaryBuiltInFunction(TERNARY_BUILT_IN_FUNCTION_EDEFAULT);
				return;
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1:
				setParameter1((Expression)null);
				return;
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2:
				setParameter2((Expression)null);
				return;
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3:
				setParameter3((Expression)null);
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
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__TERNARY_BUILT_IN_FUNCTION:
				return ternaryBuiltInFunction != TERNARY_BUILT_IN_FUNCTION_EDEFAULT;
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1:
				return parameter1 != null;
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2:
				return parameter2 != null;
			case NqcPackage.TERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3:
				return parameter3 != null;
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
		result.append(" (TernaryBuiltInFunction: ");
		result.append(ternaryBuiltInFunction);
		result.append(')');
		return result.toString();
	}

} //TernaryBuiltInFunctionCallImpl
