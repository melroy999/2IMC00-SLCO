/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import promela.PromelaPackage;
import promela.any_expr;
import promela.pc_value_expr;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>pc value expr</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.pc_value_exprImpl#getAny_expr <em>Any expr</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class pc_value_exprImpl extends any_exprImpl implements pc_value_expr {
	/**
	 * The cached value of the '{@link #getAny_expr() <em>Any expr</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAny_expr()
	 * @generated
	 * @ordered
	 */
	protected any_expr any_expr;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected pc_value_exprImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.PC_VALUE_EXPR;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public any_expr getAny_expr() {
		return any_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetAny_expr(any_expr newAny_expr, NotificationChain msgs) {
		any_expr oldAny_expr = any_expr;
		any_expr = newAny_expr;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.PC_VALUE_EXPR__ANY_EXPR, oldAny_expr, newAny_expr);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setAny_expr(any_expr newAny_expr) {
		if (newAny_expr != any_expr) {
			NotificationChain msgs = null;
			if (any_expr != null)
				msgs = ((InternalEObject)any_expr).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.PC_VALUE_EXPR__ANY_EXPR, null, msgs);
			if (newAny_expr != null)
				msgs = ((InternalEObject)newAny_expr).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.PC_VALUE_EXPR__ANY_EXPR, null, msgs);
			msgs = basicSetAny_expr(newAny_expr, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.PC_VALUE_EXPR__ANY_EXPR, newAny_expr, newAny_expr));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.PC_VALUE_EXPR__ANY_EXPR:
				return basicSetAny_expr(null, msgs);
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
			case PromelaPackage.PC_VALUE_EXPR__ANY_EXPR:
				return getAny_expr();
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
			case PromelaPackage.PC_VALUE_EXPR__ANY_EXPR:
				setAny_expr((any_expr)newValue);
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
			case PromelaPackage.PC_VALUE_EXPR__ANY_EXPR:
				setAny_expr((any_expr)null);
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
			case PromelaPackage.PC_VALUE_EXPR__ANY_EXPR:
				return any_expr != null;
		}
		return super.eIsSet(featureID);
	}

} //pc_value_exprImpl
