/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import java.util.Collection;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.util.EObjectResolvingEList;

import poosl.Expression;
import poosl.Interrupt;
import poosl.PooslPackage;
import poosl.Statement;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Interrupt</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.InterruptImpl#getBody <em>Body</em>}</li>
 *   <li>{@link poosl.impl.InterruptImpl#getCondition <em>Condition</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class InterruptImpl extends StatementImpl implements Interrupt {
	/**
	 * The cached value of the '{@link #getBody() <em>Body</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getBody()
	 * @generated
	 * @ordered
	 */
	protected EList<Statement> body;

	/**
	 * The cached value of the '{@link #getCondition() <em>Condition</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCondition()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> condition;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected InterruptImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.INTERRUPT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Statement> getBody() {
		if (body == null) {
			body = new EObjectResolvingEList<Statement>(Statement.class, this, PooslPackage.INTERRUPT__BODY);
		}
		return body;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getCondition() {
		if (condition == null) {
			condition = new EObjectResolvingEList<Expression>(Expression.class, this, PooslPackage.INTERRUPT__CONDITION);
		}
		return condition;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PooslPackage.INTERRUPT__BODY:
				return getBody();
			case PooslPackage.INTERRUPT__CONDITION:
				return getCondition();
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
			case PooslPackage.INTERRUPT__BODY:
				getBody().clear();
				getBody().addAll((Collection<? extends Statement>)newValue);
				return;
			case PooslPackage.INTERRUPT__CONDITION:
				getCondition().clear();
				getCondition().addAll((Collection<? extends Expression>)newValue);
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
			case PooslPackage.INTERRUPT__BODY:
				getBody().clear();
				return;
			case PooslPackage.INTERRUPT__CONDITION:
				getCondition().clear();
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
			case PooslPackage.INTERRUPT__BODY:
				return body != null && !body.isEmpty();
			case PooslPackage.INTERRUPT__CONDITION:
				return condition != null && !condition.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //InterruptImpl
