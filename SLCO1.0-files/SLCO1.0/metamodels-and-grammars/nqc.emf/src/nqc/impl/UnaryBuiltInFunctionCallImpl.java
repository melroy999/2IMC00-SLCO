/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.BuiltInUnaryFunctionEnum;
import nqc.Expression;
import nqc.NqcPackage;
import nqc.UnaryBuiltInFunctionCall;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Unary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.UnaryBuiltInFunctionCallImpl#getUnaryBuiltInFunction <em>Unary Built In Function</em>}</li>
 *   <li>{@link nqc.impl.UnaryBuiltInFunctionCallImpl#getParameter <em>Parameter</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class UnaryBuiltInFunctionCallImpl extends BuiltInFunctionCallImpl implements UnaryBuiltInFunctionCall {
	/**
	 * The default value of the '{@link #getUnaryBuiltInFunction() <em>Unary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUnaryBuiltInFunction()
	 * @generated
	 * @ordered
	 */
	protected static final BuiltInUnaryFunctionEnum UNARY_BUILT_IN_FUNCTION_EDEFAULT = BuiltInUnaryFunctionEnum.CLEAR_SENSOR;

	/**
	 * The cached value of the '{@link #getUnaryBuiltInFunction() <em>Unary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUnaryBuiltInFunction()
	 * @generated
	 * @ordered
	 */
	protected BuiltInUnaryFunctionEnum unaryBuiltInFunction = UNARY_BUILT_IN_FUNCTION_EDEFAULT;

	/**
	 * The cached value of the '{@link #getParameter() <em>Parameter</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getParameter()
	 * @generated
	 * @ordered
	 */
	protected Expression parameter;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected UnaryBuiltInFunctionCallImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getUnaryBuiltInFunctionCall();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BuiltInUnaryFunctionEnum getUnaryBuiltInFunction() {
		return unaryBuiltInFunction;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setUnaryBuiltInFunction(BuiltInUnaryFunctionEnum newUnaryBuiltInFunction) {
		BuiltInUnaryFunctionEnum oldUnaryBuiltInFunction = unaryBuiltInFunction;
		unaryBuiltInFunction = newUnaryBuiltInFunction == null ? UNARY_BUILT_IN_FUNCTION_EDEFAULT : newUnaryBuiltInFunction;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__UNARY_BUILT_IN_FUNCTION, oldUnaryBuiltInFunction, unaryBuiltInFunction));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getParameter() {
		return parameter;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetParameter(Expression newParameter, NotificationChain msgs) {
		Expression oldParameter = parameter;
		parameter = newParameter;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__PARAMETER, oldParameter, newParameter);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setParameter(Expression newParameter) {
		if (newParameter != parameter) {
			NotificationChain msgs = null;
			if (parameter != null)
				msgs = ((InternalEObject)parameter).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__PARAMETER, null, msgs);
			if (newParameter != null)
				msgs = ((InternalEObject)newParameter).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__PARAMETER, null, msgs);
			msgs = basicSetParameter(newParameter, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__PARAMETER, newParameter, newParameter));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__PARAMETER:
				return basicSetParameter(null, msgs);
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
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__UNARY_BUILT_IN_FUNCTION:
				return getUnaryBuiltInFunction();
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__PARAMETER:
				return getParameter();
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
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__UNARY_BUILT_IN_FUNCTION:
				setUnaryBuiltInFunction((BuiltInUnaryFunctionEnum)newValue);
				return;
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__PARAMETER:
				setParameter((Expression)newValue);
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
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__UNARY_BUILT_IN_FUNCTION:
				setUnaryBuiltInFunction(UNARY_BUILT_IN_FUNCTION_EDEFAULT);
				return;
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__PARAMETER:
				setParameter((Expression)null);
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
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__UNARY_BUILT_IN_FUNCTION:
				return unaryBuiltInFunction != UNARY_BUILT_IN_FUNCTION_EDEFAULT;
			case NqcPackage.UNARY_BUILT_IN_FUNCTION_CALL__PARAMETER:
				return parameter != null;
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
		result.append(" (UnaryBuiltInFunction: ");
		result.append(unaryBuiltInFunction);
		result.append(')');
		return result.toString();
	}

} //UnaryBuiltInFunctionCallImpl
