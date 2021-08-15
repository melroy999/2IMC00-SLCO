/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import poosl.Expression;
import poosl.LoopExpression;
import poosl.PooslPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Loop Expression</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.LoopExpressionImpl#getCondition <em>Condition</em>}</li>
 *   <li>{@link poosl.impl.LoopExpressionImpl#getBody <em>Body</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class LoopExpressionImpl extends ExpressionImpl implements LoopExpression {
	/**
	 * The cached value of the '{@link #getCondition() <em>Condition</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCondition()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> condition;

	/**
	 * The cached value of the '{@link #getBody() <em>Body</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getBody()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> body;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected LoopExpressionImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.LOOP_EXPRESSION;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getCondition() {
		if (condition == null) {
			condition = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.LOOP_EXPRESSION__CONDITION);
		}
		return condition;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getBody() {
		if (body == null) {
			body = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.LOOP_EXPRESSION__BODY);
		}
		return body;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.LOOP_EXPRESSION__CONDITION:
				return ((InternalEList<?>)getCondition()).basicRemove(otherEnd, msgs);
			case PooslPackage.LOOP_EXPRESSION__BODY:
				return ((InternalEList<?>)getBody()).basicRemove(otherEnd, msgs);
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
			case PooslPackage.LOOP_EXPRESSION__CONDITION:
				return getCondition();
			case PooslPackage.LOOP_EXPRESSION__BODY:
				return getBody();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case PooslPackage.LOOP_EXPRESSION__CONDITION:
				getCondition().clear();
				getCondition().addAll((Collection<? extends Expression>)newValue);
				return;
			case PooslPackage.LOOP_EXPRESSION__BODY:
				getBody().clear();
				getBody().addAll((Collection<? extends Expression>)newValue);
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
			case PooslPackage.LOOP_EXPRESSION__CONDITION:
				getCondition().clear();
				return;
			case PooslPackage.LOOP_EXPRESSION__BODY:
				getBody().clear();
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
			case PooslPackage.LOOP_EXPRESSION__CONDITION:
				return condition != null && !condition.isEmpty();
			case PooslPackage.LOOP_EXPRESSION__BODY:
				return body != null && !body.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //LoopExpressionImpl
