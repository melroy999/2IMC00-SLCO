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

import poosl.ConditionalExpression;
import poosl.Expression;
import poosl.PooslPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Conditional Expression</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.ConditionalExpressionImpl#getCondition <em>Condition</em>}</li>
 *   <li>{@link poosl.impl.ConditionalExpressionImpl#getThenClause <em>Then Clause</em>}</li>
 *   <li>{@link poosl.impl.ConditionalExpressionImpl#getElseClause <em>Else Clause</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ConditionalExpressionImpl extends ExpressionImpl implements ConditionalExpression {
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
	 * The cached value of the '{@link #getThenClause() <em>Then Clause</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getThenClause()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> thenClause;

	/**
	 * The cached value of the '{@link #getElseClause() <em>Else Clause</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getElseClause()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> elseClause;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ConditionalExpressionImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.CONDITIONAL_EXPRESSION;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getCondition() {
		if (condition == null) {
			condition = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.CONDITIONAL_EXPRESSION__CONDITION);
		}
		return condition;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getThenClause() {
		if (thenClause == null) {
			thenClause = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.CONDITIONAL_EXPRESSION__THEN_CLAUSE);
		}
		return thenClause;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getElseClause() {
		if (elseClause == null) {
			elseClause = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.CONDITIONAL_EXPRESSION__ELSE_CLAUSE);
		}
		return elseClause;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.CONDITIONAL_EXPRESSION__CONDITION:
				return ((InternalEList<?>)getCondition()).basicRemove(otherEnd, msgs);
			case PooslPackage.CONDITIONAL_EXPRESSION__THEN_CLAUSE:
				return ((InternalEList<?>)getThenClause()).basicRemove(otherEnd, msgs);
			case PooslPackage.CONDITIONAL_EXPRESSION__ELSE_CLAUSE:
				return ((InternalEList<?>)getElseClause()).basicRemove(otherEnd, msgs);
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
			case PooslPackage.CONDITIONAL_EXPRESSION__CONDITION:
				return getCondition();
			case PooslPackage.CONDITIONAL_EXPRESSION__THEN_CLAUSE:
				return getThenClause();
			case PooslPackage.CONDITIONAL_EXPRESSION__ELSE_CLAUSE:
				return getElseClause();
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
			case PooslPackage.CONDITIONAL_EXPRESSION__CONDITION:
				getCondition().clear();
				getCondition().addAll((Collection<? extends Expression>)newValue);
				return;
			case PooslPackage.CONDITIONAL_EXPRESSION__THEN_CLAUSE:
				getThenClause().clear();
				getThenClause().addAll((Collection<? extends Expression>)newValue);
				return;
			case PooslPackage.CONDITIONAL_EXPRESSION__ELSE_CLAUSE:
				getElseClause().clear();
				getElseClause().addAll((Collection<? extends Expression>)newValue);
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
			case PooslPackage.CONDITIONAL_EXPRESSION__CONDITION:
				getCondition().clear();
				return;
			case PooslPackage.CONDITIONAL_EXPRESSION__THEN_CLAUSE:
				getThenClause().clear();
				return;
			case PooslPackage.CONDITIONAL_EXPRESSION__ELSE_CLAUSE:
				getElseClause().clear();
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
			case PooslPackage.CONDITIONAL_EXPRESSION__CONDITION:
				return condition != null && !condition.isEmpty();
			case PooslPackage.CONDITIONAL_EXPRESSION__THEN_CLAUSE:
				return thenClause != null && !thenClause.isEmpty();
			case PooslPackage.CONDITIONAL_EXPRESSION__ELSE_CLAUSE:
				return elseClause != null && !elseClause.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //ConditionalExpressionImpl
