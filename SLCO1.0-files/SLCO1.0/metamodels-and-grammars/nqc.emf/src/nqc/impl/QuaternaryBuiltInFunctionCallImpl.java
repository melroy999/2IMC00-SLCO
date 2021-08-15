/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.BuiltInQuaternaryFunctionEnum;
import nqc.Expression;
import nqc.NqcPackage;
import nqc.QuaternaryBuiltInFunctionCall;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Quaternary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.QuaternaryBuiltInFunctionCallImpl#getQuaternaryBuiltInFunction <em>Quaternary Built In Function</em>}</li>
 *   <li>{@link nqc.impl.QuaternaryBuiltInFunctionCallImpl#getParameter1 <em>Parameter1</em>}</li>
 *   <li>{@link nqc.impl.QuaternaryBuiltInFunctionCallImpl#getParameter2 <em>Parameter2</em>}</li>
 *   <li>{@link nqc.impl.QuaternaryBuiltInFunctionCallImpl#getParameter3 <em>Parameter3</em>}</li>
 *   <li>{@link nqc.impl.QuaternaryBuiltInFunctionCallImpl#getParameter4 <em>Parameter4</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class QuaternaryBuiltInFunctionCallImpl extends BuiltInFunctionCallImpl implements QuaternaryBuiltInFunctionCall {
	/**
	 * The default value of the '{@link #getQuaternaryBuiltInFunction() <em>Quaternary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getQuaternaryBuiltInFunction()
	 * @generated
	 * @ordered
	 */
	protected static final BuiltInQuaternaryFunctionEnum QUATERNARY_BUILT_IN_FUNCTION_EDEFAULT = BuiltInQuaternaryFunctionEnum.CALIBRATE_EVENT;

	/**
	 * The cached value of the '{@link #getQuaternaryBuiltInFunction() <em>Quaternary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getQuaternaryBuiltInFunction()
	 * @generated
	 * @ordered
	 */
	protected BuiltInQuaternaryFunctionEnum quaternaryBuiltInFunction = QUATERNARY_BUILT_IN_FUNCTION_EDEFAULT;

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
	 * The cached value of the '{@link #getParameter4() <em>Parameter4</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getParameter4()
	 * @generated
	 * @ordered
	 */
	protected Expression parameter4;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected QuaternaryBuiltInFunctionCallImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getQuaternaryBuiltInFunctionCall();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInQuaternaryFunctionEnum getQuaternaryBuiltInFunction() {
		return quaternaryBuiltInFunction;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setQuaternaryBuiltInFunction(BuiltInQuaternaryFunctionEnum newQuaternaryBuiltInFunction) {
		BuiltInQuaternaryFunctionEnum oldQuaternaryBuiltInFunction = quaternaryBuiltInFunction;
		quaternaryBuiltInFunction = newQuaternaryBuiltInFunction == null ? QUATERNARY_BUILT_IN_FUNCTION_EDEFAULT : newQuaternaryBuiltInFunction;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__QUATERNARY_BUILT_IN_FUNCTION, oldQuaternaryBuiltInFunction, quaternaryBuiltInFunction));
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1, oldParameter1, newParameter1);
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
				msgs = ((InternalEObject)parameter1).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1, null, msgs);
			if (newParameter1 != null)
				msgs = ((InternalEObject)newParameter1).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1, null, msgs);
			msgs = basicSetParameter1(newParameter1, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1, newParameter1, newParameter1));
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2, oldParameter2, newParameter2);
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
				msgs = ((InternalEObject)parameter2).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2, null, msgs);
			if (newParameter2 != null)
				msgs = ((InternalEObject)newParameter2).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2, null, msgs);
			msgs = basicSetParameter2(newParameter2, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2, newParameter2, newParameter2));
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3, oldParameter3, newParameter3);
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
				msgs = ((InternalEObject)parameter3).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3, null, msgs);
			if (newParameter3 != null)
				msgs = ((InternalEObject)newParameter3).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3, null, msgs);
			msgs = basicSetParameter3(newParameter3, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3, newParameter3, newParameter3));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getParameter4() {
		return parameter4;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetParameter4(Expression newParameter4, NotificationChain msgs) {
		Expression oldParameter4 = parameter4;
		parameter4 = newParameter4;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER4, oldParameter4, newParameter4);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setParameter4(Expression newParameter4) {
		if (newParameter4 != parameter4) {
			NotificationChain msgs = null;
			if (parameter4 != null)
				msgs = ((InternalEObject)parameter4).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER4, null, msgs);
			if (newParameter4 != null)
				msgs = ((InternalEObject)newParameter4).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER4, null, msgs);
			msgs = basicSetParameter4(newParameter4, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER4, newParameter4, newParameter4));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1:
				return basicSetParameter1(null, msgs);
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2:
				return basicSetParameter2(null, msgs);
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3:
				return basicSetParameter3(null, msgs);
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER4:
				return basicSetParameter4(null, msgs);
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
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__QUATERNARY_BUILT_IN_FUNCTION:
				return getQuaternaryBuiltInFunction();
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1:
				return getParameter1();
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2:
				return getParameter2();
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3:
				return getParameter3();
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER4:
				return getParameter4();
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
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__QUATERNARY_BUILT_IN_FUNCTION:
				setQuaternaryBuiltInFunction((BuiltInQuaternaryFunctionEnum)newValue);
				return;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1:
				setParameter1((Expression)newValue);
				return;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2:
				setParameter2((Expression)newValue);
				return;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3:
				setParameter3((Expression)newValue);
				return;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER4:
				setParameter4((Expression)newValue);
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
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__QUATERNARY_BUILT_IN_FUNCTION:
				setQuaternaryBuiltInFunction(QUATERNARY_BUILT_IN_FUNCTION_EDEFAULT);
				return;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1:
				setParameter1((Expression)null);
				return;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2:
				setParameter2((Expression)null);
				return;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3:
				setParameter3((Expression)null);
				return;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER4:
				setParameter4((Expression)null);
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
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__QUATERNARY_BUILT_IN_FUNCTION:
				return quaternaryBuiltInFunction != QUATERNARY_BUILT_IN_FUNCTION_EDEFAULT;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER1:
				return parameter1 != null;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER2:
				return parameter2 != null;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER3:
				return parameter3 != null;
			case NqcPackage.QUATERNARY_BUILT_IN_FUNCTION_CALL__PARAMETER4:
				return parameter4 != null;
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
		result.append(" (QuaternaryBuiltInFunction: ");
		result.append(quaternaryBuiltInFunction);
		result.append(')');
		return result.toString();
	}

} //QuaternaryBuiltInFunctionCallImpl
