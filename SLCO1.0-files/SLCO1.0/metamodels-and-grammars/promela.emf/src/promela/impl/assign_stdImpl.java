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
import promela.assign_std;
import promela.varref;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>assign std</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.assign_stdImpl#getVarref <em>Varref</em>}</li>
 *   <li>{@link promela.impl.assign_stdImpl#getAny_expr <em>Any expr</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class assign_stdImpl extends assignImpl implements assign_std {
	/**
	 * The cached value of the '{@link #getVarref() <em>Varref</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVarref()
	 * @generated
	 * @ordered
	 */
	protected varref varref;

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
	protected assign_stdImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.ASSIGN_STD;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public varref getVarref() {
		return varref;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetVarref(varref newVarref, NotificationChain msgs) {
		varref oldVarref = varref;
		varref = newVarref;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.ASSIGN_STD__VARREF, oldVarref, newVarref);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setVarref(varref newVarref) {
		if (newVarref != varref) {
			NotificationChain msgs = null;
			if (varref != null)
				msgs = ((InternalEObject)varref).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.ASSIGN_STD__VARREF, null, msgs);
			if (newVarref != null)
				msgs = ((InternalEObject)newVarref).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.ASSIGN_STD__VARREF, null, msgs);
			msgs = basicSetVarref(newVarref, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.ASSIGN_STD__VARREF, newVarref, newVarref));
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.ASSIGN_STD__ANY_EXPR, oldAny_expr, newAny_expr);
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
				msgs = ((InternalEObject)any_expr).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.ASSIGN_STD__ANY_EXPR, null, msgs);
			if (newAny_expr != null)
				msgs = ((InternalEObject)newAny_expr).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.ASSIGN_STD__ANY_EXPR, null, msgs);
			msgs = basicSetAny_expr(newAny_expr, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.ASSIGN_STD__ANY_EXPR, newAny_expr, newAny_expr));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.ASSIGN_STD__VARREF:
				return basicSetVarref(null, msgs);
			case PromelaPackage.ASSIGN_STD__ANY_EXPR:
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
			case PromelaPackage.ASSIGN_STD__VARREF:
				return getVarref();
			case PromelaPackage.ASSIGN_STD__ANY_EXPR:
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
			case PromelaPackage.ASSIGN_STD__VARREF:
				setVarref((varref)newValue);
				return;
			case PromelaPackage.ASSIGN_STD__ANY_EXPR:
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
			case PromelaPackage.ASSIGN_STD__VARREF:
				setVarref((varref)null);
				return;
			case PromelaPackage.ASSIGN_STD__ANY_EXPR:
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
			case PromelaPackage.ASSIGN_STD__VARREF:
				return varref != null;
			case PromelaPackage.ASSIGN_STD__ANY_EXPR:
				return any_expr != null;
		}
		return super.eIsSet(featureID);
	}

} //assign_stdImpl
