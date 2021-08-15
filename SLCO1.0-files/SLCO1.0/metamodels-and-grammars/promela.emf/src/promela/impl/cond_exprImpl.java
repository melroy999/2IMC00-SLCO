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
import promela.cond_expr;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>cond expr</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.cond_exprImpl#getExpr1 <em>Expr1</em>}</li>
 *   <li>{@link promela.impl.cond_exprImpl#getExpr2 <em>Expr2</em>}</li>
 *   <li>{@link promela.impl.cond_exprImpl#getExpr3 <em>Expr3</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class cond_exprImpl extends any_exprImpl implements cond_expr {
	/**
	 * The cached value of the '{@link #getExpr1() <em>Expr1</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getExpr1()
	 * @generated
	 * @ordered
	 */
	protected any_expr expr1;

	/**
	 * The cached value of the '{@link #getExpr2() <em>Expr2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getExpr2()
	 * @generated
	 * @ordered
	 */
	protected any_expr expr2;

	/**
	 * The cached value of the '{@link #getExpr3() <em>Expr3</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getExpr3()
	 * @generated
	 * @ordered
	 */
	protected any_expr expr3;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected cond_exprImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.COND_EXPR;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public any_expr getExpr1() {
		return expr1;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetExpr1(any_expr newExpr1, NotificationChain msgs) {
		any_expr oldExpr1 = expr1;
		expr1 = newExpr1;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.COND_EXPR__EXPR1, oldExpr1, newExpr1);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setExpr1(any_expr newExpr1) {
		if (newExpr1 != expr1) {
			NotificationChain msgs = null;
			if (expr1 != null)
				msgs = ((InternalEObject)expr1).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.COND_EXPR__EXPR1, null, msgs);
			if (newExpr1 != null)
				msgs = ((InternalEObject)newExpr1).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.COND_EXPR__EXPR1, null, msgs);
			msgs = basicSetExpr1(newExpr1, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.COND_EXPR__EXPR1, newExpr1, newExpr1));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public any_expr getExpr2() {
		return expr2;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetExpr2(any_expr newExpr2, NotificationChain msgs) {
		any_expr oldExpr2 = expr2;
		expr2 = newExpr2;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.COND_EXPR__EXPR2, oldExpr2, newExpr2);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setExpr2(any_expr newExpr2) {
		if (newExpr2 != expr2) {
			NotificationChain msgs = null;
			if (expr2 != null)
				msgs = ((InternalEObject)expr2).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.COND_EXPR__EXPR2, null, msgs);
			if (newExpr2 != null)
				msgs = ((InternalEObject)newExpr2).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.COND_EXPR__EXPR2, null, msgs);
			msgs = basicSetExpr2(newExpr2, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.COND_EXPR__EXPR2, newExpr2, newExpr2));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public any_expr getExpr3() {
		return expr3;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetExpr3(any_expr newExpr3, NotificationChain msgs) {
		any_expr oldExpr3 = expr3;
		expr3 = newExpr3;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.COND_EXPR__EXPR3, oldExpr3, newExpr3);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setExpr3(any_expr newExpr3) {
		if (newExpr3 != expr3) {
			NotificationChain msgs = null;
			if (expr3 != null)
				msgs = ((InternalEObject)expr3).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.COND_EXPR__EXPR3, null, msgs);
			if (newExpr3 != null)
				msgs = ((InternalEObject)newExpr3).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.COND_EXPR__EXPR3, null, msgs);
			msgs = basicSetExpr3(newExpr3, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.COND_EXPR__EXPR3, newExpr3, newExpr3));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.COND_EXPR__EXPR1:
				return basicSetExpr1(null, msgs);
			case PromelaPackage.COND_EXPR__EXPR2:
				return basicSetExpr2(null, msgs);
			case PromelaPackage.COND_EXPR__EXPR3:
				return basicSetExpr3(null, msgs);
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
			case PromelaPackage.COND_EXPR__EXPR1:
				return getExpr1();
			case PromelaPackage.COND_EXPR__EXPR2:
				return getExpr2();
			case PromelaPackage.COND_EXPR__EXPR3:
				return getExpr3();
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
			case PromelaPackage.COND_EXPR__EXPR1:
				setExpr1((any_expr)newValue);
				return;
			case PromelaPackage.COND_EXPR__EXPR2:
				setExpr2((any_expr)newValue);
				return;
			case PromelaPackage.COND_EXPR__EXPR3:
				setExpr3((any_expr)newValue);
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
			case PromelaPackage.COND_EXPR__EXPR1:
				setExpr1((any_expr)null);
				return;
			case PromelaPackage.COND_EXPR__EXPR2:
				setExpr2((any_expr)null);
				return;
			case PromelaPackage.COND_EXPR__EXPR3:
				setExpr3((any_expr)null);
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
			case PromelaPackage.COND_EXPR__EXPR1:
				return expr1 != null;
			case PromelaPackage.COND_EXPR__EXPR2:
				return expr2 != null;
			case PromelaPackage.COND_EXPR__EXPR3:
				return expr3 != null;
		}
		return super.eIsSet(featureID);
	}

} //cond_exprImpl
