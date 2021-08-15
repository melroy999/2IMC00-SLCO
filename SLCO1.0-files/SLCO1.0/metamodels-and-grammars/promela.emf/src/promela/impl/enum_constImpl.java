/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import promela.PromelaPackage;
import promela.constenum;
import promela.enum_const;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>enum const</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.enum_constImpl#getConst_enum <em>Const enum</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class enum_constImpl extends constantImpl implements enum_const {
	/**
	 * The default value of the '{@link #getConst_enum() <em>Const enum</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getConst_enum()
	 * @generated
	 * @ordered
	 */
	protected static final constenum CONST_ENUM_EDEFAULT = constenum.TRUE;

	/**
	 * The cached value of the '{@link #getConst_enum() <em>Const enum</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getConst_enum()
	 * @generated
	 * @ordered
	 */
	protected constenum const_enum = CONST_ENUM_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected enum_constImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.ENUM_CONST;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public constenum getConst_enum() {
		return const_enum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setConst_enum(constenum newConst_enum) {
		constenum oldConst_enum = const_enum;
		const_enum = newConst_enum == null ? CONST_ENUM_EDEFAULT : newConst_enum;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.ENUM_CONST__CONST_ENUM, oldConst_enum, const_enum));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.ENUM_CONST__CONST_ENUM:
				return getConst_enum();
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
			case PromelaPackage.ENUM_CONST__CONST_ENUM:
				setConst_enum((constenum)newValue);
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
			case PromelaPackage.ENUM_CONST__CONST_ENUM:
				setConst_enum(CONST_ENUM_EDEFAULT);
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
			case PromelaPackage.ENUM_CONST__CONST_ENUM:
				return const_enum != CONST_ENUM_EDEFAULT;
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
		result.append(" (const_enum: ");
		result.append(const_enum);
		result.append(')');
		return result.toString();
	}

} //enum_constImpl
