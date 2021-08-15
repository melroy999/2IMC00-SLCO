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
import poosl.PooslPackage;
import poosl.ReturnExpression;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Return Expression</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.ReturnExpressionImpl#getReturnExpressions <em>Return Expressions</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ReturnExpressionImpl extends ExpressionImpl implements ReturnExpression {
	/**
	 * The cached value of the '{@link #getReturnExpressions() <em>Return Expressions</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReturnExpressions()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> returnExpressions;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ReturnExpressionImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.RETURN_EXPRESSION;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getReturnExpressions() {
		if (returnExpressions == null) {
			returnExpressions = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.RETURN_EXPRESSION__RETURN_EXPRESSIONS);
		}
		return returnExpressions;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.RETURN_EXPRESSION__RETURN_EXPRESSIONS:
				return ((InternalEList<?>)getReturnExpressions()).basicRemove(otherEnd, msgs);
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
			case PooslPackage.RETURN_EXPRESSION__RETURN_EXPRESSIONS:
				return getReturnExpressions();
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
			case PooslPackage.RETURN_EXPRESSION__RETURN_EXPRESSIONS:
				getReturnExpressions().clear();
				getReturnExpressions().addAll((Collection<? extends Expression>)newValue);
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
			case PooslPackage.RETURN_EXPRESSION__RETURN_EXPRESSIONS:
				getReturnExpressions().clear();
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
			case PooslPackage.RETURN_EXPRESSION__RETURN_EXPRESSIONS:
				return returnExpressions != null && !returnExpressions.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //ReturnExpressionImpl
