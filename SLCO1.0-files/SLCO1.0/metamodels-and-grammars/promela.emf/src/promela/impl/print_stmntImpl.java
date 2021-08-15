/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import promela.PromelaPackage;
import promela.any_expr;
import promela.print_stmnt;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>print stmnt</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.print_stmntImpl#getString <em>String</em>}</li>
 *   <li>{@link promela.impl.print_stmntImpl#getArg_lst <em>Arg lst</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class print_stmntImpl extends stmntImpl implements print_stmnt {
	/**
	 * The default value of the '{@link #getString() <em>String</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getString()
	 * @generated
	 * @ordered
	 */
	protected static final String STRING_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getString() <em>String</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getString()
	 * @generated
	 * @ordered
	 */
	protected String string = STRING_EDEFAULT;

	/**
	 * The cached value of the '{@link #getArg_lst() <em>Arg lst</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getArg_lst()
	 * @generated
	 * @ordered
	 */
	protected EList<any_expr> arg_lst;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected print_stmntImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.PRINT_STMNT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getString() {
		return string;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setString(String newString) {
		String oldString = string;
		string = newString;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.PRINT_STMNT__STRING, oldString, string));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<any_expr> getArg_lst() {
		if (arg_lst == null) {
			arg_lst = new EObjectContainmentEList<any_expr>(any_expr.class, this, PromelaPackage.PRINT_STMNT__ARG_LST);
		}
		return arg_lst;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.PRINT_STMNT__ARG_LST:
				return ((InternalEList<?>)getArg_lst()).basicRemove(otherEnd, msgs);
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
			case PromelaPackage.PRINT_STMNT__STRING:
				return getString();
			case PromelaPackage.PRINT_STMNT__ARG_LST:
				return getArg_lst();
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
			case PromelaPackage.PRINT_STMNT__STRING:
				setString((String)newValue);
				return;
			case PromelaPackage.PRINT_STMNT__ARG_LST:
				getArg_lst().clear();
				getArg_lst().addAll((Collection<? extends any_expr>)newValue);
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
			case PromelaPackage.PRINT_STMNT__STRING:
				setString(STRING_EDEFAULT);
				return;
			case PromelaPackage.PRINT_STMNT__ARG_LST:
				getArg_lst().clear();
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
			case PromelaPackage.PRINT_STMNT__STRING:
				return STRING_EDEFAULT == null ? string != null : !STRING_EDEFAULT.equals(string);
			case PromelaPackage.PRINT_STMNT__ARG_LST:
				return arg_lst != null && !arg_lst.isEmpty();
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
		result.append(" (string: ");
		result.append(string);
		result.append(')');
		return result.toString();
	}

} //print_stmntImpl
