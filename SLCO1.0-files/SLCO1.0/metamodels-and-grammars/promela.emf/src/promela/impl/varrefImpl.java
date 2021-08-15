/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import promela.PromelaPackage;
import promela.any_expr;
import promela.varref;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>varref</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.varrefImpl#getVar_name <em>Var name</em>}</li>
 *   <li>{@link promela.impl.varrefImpl#getAny_expr <em>Any expr</em>}</li>
 *   <li>{@link promela.impl.varrefImpl#getVarref <em>Varref</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class varrefImpl extends any_exprImpl implements varref {
	/**
	 * The default value of the '{@link #getVar_name() <em>Var name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVar_name()
	 * @generated
	 * @ordered
	 */
	protected static final String VAR_NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getVar_name() <em>Var name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVar_name()
	 * @generated
	 * @ordered
	 */
	protected String var_name = VAR_NAME_EDEFAULT;

	/**
	 * The cached value of the '{@link #getAny_expr() <em>Any expr</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAny_expr()
	 * @generated
	 * @ordered
	 */
	protected any_expr any_expr;

	/**
	 * The cached value of the '{@link #getVarref() <em>Varref</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVarref()
	 * @generated
	 * @ordered
	 */
	protected varref varref;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected varrefImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.VARREF;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getVar_name() {
		return var_name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setVar_name(String newVar_name) {
		String oldVar_name = var_name;
		var_name = newVar_name;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.VARREF__VAR_NAME, oldVar_name, var_name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public any_expr getAny_expr() {
		if (any_expr != null && any_expr.eIsProxy()) {
			InternalEObject oldAny_expr = (InternalEObject)any_expr;
			any_expr = (any_expr)eResolveProxy(oldAny_expr);
			if (any_expr != oldAny_expr) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PromelaPackage.VARREF__ANY_EXPR, oldAny_expr, any_expr));
			}
		}
		return any_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public any_expr basicGetAny_expr() {
		return any_expr;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setAny_expr(any_expr newAny_expr) {
		any_expr oldAny_expr = any_expr;
		any_expr = newAny_expr;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.VARREF__ANY_EXPR, oldAny_expr, any_expr));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public varref getVarref() {
		if (varref != null && varref.eIsProxy()) {
			InternalEObject oldVarref = (InternalEObject)varref;
			varref = (varref)eResolveProxy(oldVarref);
			if (varref != oldVarref) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PromelaPackage.VARREF__VARREF, oldVarref, varref));
			}
		}
		return varref;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public varref basicGetVarref() {
		return varref;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setVarref(varref newVarref) {
		varref oldVarref = varref;
		varref = newVarref;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.VARREF__VARREF, oldVarref, varref));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.VARREF__VAR_NAME:
				return getVar_name();
			case PromelaPackage.VARREF__ANY_EXPR:
				if (resolve) return getAny_expr();
				return basicGetAny_expr();
			case PromelaPackage.VARREF__VARREF:
				if (resolve) return getVarref();
				return basicGetVarref();
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
			case PromelaPackage.VARREF__VAR_NAME:
				setVar_name((String)newValue);
				return;
			case PromelaPackage.VARREF__ANY_EXPR:
				setAny_expr((any_expr)newValue);
				return;
			case PromelaPackage.VARREF__VARREF:
				setVarref((varref)newValue);
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
			case PromelaPackage.VARREF__VAR_NAME:
				setVar_name(VAR_NAME_EDEFAULT);
				return;
			case PromelaPackage.VARREF__ANY_EXPR:
				setAny_expr((any_expr)null);
				return;
			case PromelaPackage.VARREF__VARREF:
				setVarref((varref)null);
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
			case PromelaPackage.VARREF__VAR_NAME:
				return VAR_NAME_EDEFAULT == null ? var_name != null : !VAR_NAME_EDEFAULT.equals(var_name);
			case PromelaPackage.VARREF__ANY_EXPR:
				return any_expr != null;
			case PromelaPackage.VARREF__VARREF:
				return varref != null;
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
		result.append(" (var_name: ");
		result.append(var_name);
		result.append(')');
		return result.toString();
	}

} //varrefImpl
