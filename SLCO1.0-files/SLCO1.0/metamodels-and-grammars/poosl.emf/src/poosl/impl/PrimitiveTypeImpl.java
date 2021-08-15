/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import poosl.PooslPackage;
import poosl.PrimitiveType;
import poosl.PrimitiveTypeEnum;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Primitive Type</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.PrimitiveTypeImpl#getPrimitiveTypeEnum <em>Primitive Type Enum</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class PrimitiveTypeImpl extends TypeImpl implements PrimitiveType {
	/**
	 * The default value of the '{@link #getPrimitiveTypeEnum() <em>Primitive Type Enum</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPrimitiveTypeEnum()
	 * @generated
	 * @ordered
	 */
	protected static final PrimitiveTypeEnum PRIMITIVE_TYPE_ENUM_EDEFAULT = PrimitiveTypeEnum.STRING;

	/**
	 * The cached value of the '{@link #getPrimitiveTypeEnum() <em>Primitive Type Enum</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPrimitiveTypeEnum()
	 * @generated
	 * @ordered
	 */
	protected PrimitiveTypeEnum primitiveTypeEnum = PRIMITIVE_TYPE_ENUM_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected PrimitiveTypeImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.PRIMITIVE_TYPE;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PrimitiveTypeEnum getPrimitiveTypeEnum() {
		return primitiveTypeEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPrimitiveTypeEnum(PrimitiveTypeEnum newPrimitiveTypeEnum) {
		PrimitiveTypeEnum oldPrimitiveTypeEnum = primitiveTypeEnum;
		primitiveTypeEnum = newPrimitiveTypeEnum == null ? PRIMITIVE_TYPE_ENUM_EDEFAULT : newPrimitiveTypeEnum;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.PRIMITIVE_TYPE__PRIMITIVE_TYPE_ENUM, oldPrimitiveTypeEnum, primitiveTypeEnum));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PooslPackage.PRIMITIVE_TYPE__PRIMITIVE_TYPE_ENUM:
				return getPrimitiveTypeEnum();
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
			case PooslPackage.PRIMITIVE_TYPE__PRIMITIVE_TYPE_ENUM:
				setPrimitiveTypeEnum((PrimitiveTypeEnum)newValue);
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
			case PooslPackage.PRIMITIVE_TYPE__PRIMITIVE_TYPE_ENUM:
				setPrimitiveTypeEnum(PRIMITIVE_TYPE_ENUM_EDEFAULT);
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
			case PooslPackage.PRIMITIVE_TYPE__PRIMITIVE_TYPE_ENUM:
				return primitiveTypeEnum != PRIMITIVE_TYPE_ENUM_EDEFAULT;
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
		result.append(" (primitiveTypeEnum: ");
		result.append(primitiveTypeEnum);
		result.append(')');
		return result.toString();
	}

} //PrimitiveTypeImpl
