/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.Expression;
import nqc.NqcPackage;
import nqc.TernaryExpression;
import nqc.TernaryOperatorEnum;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Ternary Expression</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.TernaryExpressionImpl#getOperator <em>Operator</em>}</li>
 *   <li>{@link nqc.impl.TernaryExpressionImpl#getOperand1 <em>Operand1</em>}</li>
 *   <li>{@link nqc.impl.TernaryExpressionImpl#getOperand2 <em>Operand2</em>}</li>
 *   <li>{@link nqc.impl.TernaryExpressionImpl#getOperand3 <em>Operand3</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class TernaryExpressionImpl extends CompoundExpressionImpl implements TernaryExpression {
	/**
	 * The default value of the '{@link #getOperator() <em>Operator</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOperator()
	 * @generated
	 * @ordered
	 */
	protected static final TernaryOperatorEnum OPERATOR_EDEFAULT = TernaryOperatorEnum.COND;

	/**
	 * The cached value of the '{@link #getOperator() <em>Operator</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOperator()
	 * @generated
	 * @ordered
	 */
	protected TernaryOperatorEnum operator = OPERATOR_EDEFAULT;

	/**
	 * The cached value of the '{@link #getOperand1() <em>Operand1</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOperand1()
	 * @generated
	 * @ordered
	 */
	protected Expression operand1;

	/**
	 * The cached value of the '{@link #getOperand2() <em>Operand2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOperand2()
	 * @generated
	 * @ordered
	 */
	protected Expression operand2;

	/**
	 * The cached value of the '{@link #getOperand3() <em>Operand3</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOperand3()
	 * @generated
	 * @ordered
	 */
	protected Expression operand3;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected TernaryExpressionImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getTernaryExpression();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TernaryOperatorEnum getOperator() {
		return operator;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setOperator(TernaryOperatorEnum newOperator) {
		TernaryOperatorEnum oldOperator = operator;
		operator = newOperator == null ? OPERATOR_EDEFAULT : newOperator;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_EXPRESSION__OPERATOR, oldOperator, operator));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getOperand1() {
		return operand1;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetOperand1(Expression newOperand1, NotificationChain msgs) {
		Expression oldOperand1 = operand1;
		operand1 = newOperand1;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_EXPRESSION__OPERAND1, oldOperand1, newOperand1);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setOperand1(Expression newOperand1) {
		if (newOperand1 != operand1) {
			NotificationChain msgs = null;
			if (operand1 != null)
				msgs = ((InternalEObject)operand1).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_EXPRESSION__OPERAND1, null, msgs);
			if (newOperand1 != null)
				msgs = ((InternalEObject)newOperand1).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_EXPRESSION__OPERAND1, null, msgs);
			msgs = basicSetOperand1(newOperand1, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_EXPRESSION__OPERAND1, newOperand1, newOperand1));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getOperand2() {
		return operand2;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetOperand2(Expression newOperand2, NotificationChain msgs) {
		Expression oldOperand2 = operand2;
		operand2 = newOperand2;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_EXPRESSION__OPERAND2, oldOperand2, newOperand2);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setOperand2(Expression newOperand2) {
		if (newOperand2 != operand2) {
			NotificationChain msgs = null;
			if (operand2 != null)
				msgs = ((InternalEObject)operand2).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_EXPRESSION__OPERAND2, null, msgs);
			if (newOperand2 != null)
				msgs = ((InternalEObject)newOperand2).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_EXPRESSION__OPERAND2, null, msgs);
			msgs = basicSetOperand2(newOperand2, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_EXPRESSION__OPERAND2, newOperand2, newOperand2));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getOperand3() {
		return operand3;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetOperand3(Expression newOperand3, NotificationChain msgs) {
		Expression oldOperand3 = operand3;
		operand3 = newOperand3;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_EXPRESSION__OPERAND3, oldOperand3, newOperand3);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setOperand3(Expression newOperand3) {
		if (newOperand3 != operand3) {
			NotificationChain msgs = null;
			if (operand3 != null)
				msgs = ((InternalEObject)operand3).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_EXPRESSION__OPERAND3, null, msgs);
			if (newOperand3 != null)
				msgs = ((InternalEObject)newOperand3).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.TERNARY_EXPRESSION__OPERAND3, null, msgs);
			msgs = basicSetOperand3(newOperand3, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.TERNARY_EXPRESSION__OPERAND3, newOperand3, newOperand3));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.TERNARY_EXPRESSION__OPERAND1:
				return basicSetOperand1(null, msgs);
			case NqcPackage.TERNARY_EXPRESSION__OPERAND2:
				return basicSetOperand2(null, msgs);
			case NqcPackage.TERNARY_EXPRESSION__OPERAND3:
				return basicSetOperand3(null, msgs);
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
			case NqcPackage.TERNARY_EXPRESSION__OPERATOR:
				return getOperator();
			case NqcPackage.TERNARY_EXPRESSION__OPERAND1:
				return getOperand1();
			case NqcPackage.TERNARY_EXPRESSION__OPERAND2:
				return getOperand2();
			case NqcPackage.TERNARY_EXPRESSION__OPERAND3:
				return getOperand3();
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
			case NqcPackage.TERNARY_EXPRESSION__OPERATOR:
				setOperator((TernaryOperatorEnum)newValue);
				return;
			case NqcPackage.TERNARY_EXPRESSION__OPERAND1:
				setOperand1((Expression)newValue);
				return;
			case NqcPackage.TERNARY_EXPRESSION__OPERAND2:
				setOperand2((Expression)newValue);
				return;
			case NqcPackage.TERNARY_EXPRESSION__OPERAND3:
				setOperand3((Expression)newValue);
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
			case NqcPackage.TERNARY_EXPRESSION__OPERATOR:
				setOperator(OPERATOR_EDEFAULT);
				return;
			case NqcPackage.TERNARY_EXPRESSION__OPERAND1:
				setOperand1((Expression)null);
				return;
			case NqcPackage.TERNARY_EXPRESSION__OPERAND2:
				setOperand2((Expression)null);
				return;
			case NqcPackage.TERNARY_EXPRESSION__OPERAND3:
				setOperand3((Expression)null);
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
			case NqcPackage.TERNARY_EXPRESSION__OPERATOR:
				return operator != OPERATOR_EDEFAULT;
			case NqcPackage.TERNARY_EXPRESSION__OPERAND1:
				return operand1 != null;
			case NqcPackage.TERNARY_EXPRESSION__OPERAND2:
				return operand2 != null;
			case NqcPackage.TERNARY_EXPRESSION__OPERAND3:
				return operand3 != null;
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
		result.append(" (Operator: ");
		result.append(operator);
		result.append(')');
		return result.toString();
	}

} //TernaryExpressionImpl
