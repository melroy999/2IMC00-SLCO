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
import promela.name_expr;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>name expr</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.name_exprImpl#getName <em>Name</em>}</li>
 *   <li>{@link promela.impl.name_exprImpl#getAny_expr <em>Any expr</em>}</li>
 *   <li>{@link promela.impl.name_exprImpl#getAt_name <em>At name</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class name_exprImpl extends any_exprImpl implements name_expr {
	/**
	 * The default value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected static final String NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected String name = NAME_EDEFAULT;

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
	 * The default value of the '{@link #getAt_name() <em>At name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAt_name()
	 * @generated
	 * @ordered
	 */
	protected static final String AT_NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getAt_name() <em>At name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAt_name()
	 * @generated
	 * @ordered
	 */
	protected String at_name = AT_NAME_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected name_exprImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.NAME_EXPR;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
		return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setName(String newName) {
		String oldName = name;
		name = newName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.NAME_EXPR__NAME, oldName, name));
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.NAME_EXPR__ANY_EXPR, oldAny_expr, newAny_expr);
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
				msgs = ((InternalEObject)any_expr).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.NAME_EXPR__ANY_EXPR, null, msgs);
			if (newAny_expr != null)
				msgs = ((InternalEObject)newAny_expr).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.NAME_EXPR__ANY_EXPR, null, msgs);
			msgs = basicSetAny_expr(newAny_expr, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.NAME_EXPR__ANY_EXPR, newAny_expr, newAny_expr));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getAt_name() {
		return at_name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setAt_name(String newAt_name) {
		String oldAt_name = at_name;
		at_name = newAt_name;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.NAME_EXPR__AT_NAME, oldAt_name, at_name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.NAME_EXPR__ANY_EXPR:
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
			case PromelaPackage.NAME_EXPR__NAME:
				return getName();
			case PromelaPackage.NAME_EXPR__ANY_EXPR:
				return getAny_expr();
			case PromelaPackage.NAME_EXPR__AT_NAME:
				return getAt_name();
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
			case PromelaPackage.NAME_EXPR__NAME:
				setName((String)newValue);
				return;
			case PromelaPackage.NAME_EXPR__ANY_EXPR:
				setAny_expr((any_expr)newValue);
				return;
			case PromelaPackage.NAME_EXPR__AT_NAME:
				setAt_name((String)newValue);
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
			case PromelaPackage.NAME_EXPR__NAME:
				setName(NAME_EDEFAULT);
				return;
			case PromelaPackage.NAME_EXPR__ANY_EXPR:
				setAny_expr((any_expr)null);
				return;
			case PromelaPackage.NAME_EXPR__AT_NAME:
				setAt_name(AT_NAME_EDEFAULT);
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
			case PromelaPackage.NAME_EXPR__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case PromelaPackage.NAME_EXPR__ANY_EXPR:
				return any_expr != null;
			case PromelaPackage.NAME_EXPR__AT_NAME:
				return AT_NAME_EDEFAULT == null ? at_name != null : !AT_NAME_EDEFAULT.equals(at_name);
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
		result.append(" (name: ");
		result.append(name);
		result.append(", at_name: ");
		result.append(at_name);
		result.append(')');
		return result.toString();
	}

} //name_exprImpl
