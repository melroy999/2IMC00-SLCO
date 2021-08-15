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
import org.eclipse.emf.ecore.impl.EObjectImpl;

import poosl.Constant;
import poosl.PooslPackage;
import poosl.PrimitiveTypeEnum;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.ConstantImpl#getInteger <em>Integer</em>}</li>
 *   <li>{@link poosl.impl.ConstantImpl#getString <em>String</em>}</li>
 *   <li>{@link poosl.impl.ConstantImpl#isBoolean <em>Boolean</em>}</li>
 *   <li>{@link poosl.impl.ConstantImpl#getReal <em>Real</em>}</li>
 *   <li>{@link poosl.impl.ConstantImpl#getType <em>Type</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ConstantImpl extends EObjectImpl implements Constant {
	/**
	 * The default value of the '{@link #getInteger() <em>Integer</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInteger()
	 * @generated
	 * @ordered
	 */
	protected static final int INTEGER_EDEFAULT = 0;

	/**
	 * The cached value of the '{@link #getInteger() <em>Integer</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInteger()
	 * @generated
	 * @ordered
	 */
	protected int integer = INTEGER_EDEFAULT;

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
	 * The default value of the '{@link #isBoolean() <em>Boolean</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isBoolean()
	 * @generated
	 * @ordered
	 */
	protected static final boolean BOOLEAN_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isBoolean() <em>Boolean</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isBoolean()
	 * @generated
	 * @ordered
	 */
	protected boolean boolean_ = BOOLEAN_EDEFAULT;

	/**
	 * The default value of the '{@link #getReal() <em>Real</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReal()
	 * @generated
	 * @ordered
	 */
	protected static final float REAL_EDEFAULT = 0.0F;

	/**
	 * The cached value of the '{@link #getReal() <em>Real</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReal()
	 * @generated
	 * @ordered
	 */
	protected float real = REAL_EDEFAULT;

	/**
	 * The default value of the '{@link #getType() <em>Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getType()
	 * @generated
	 * @ordered
	 */
	protected static final PrimitiveTypeEnum TYPE_EDEFAULT = PrimitiveTypeEnum.STRING;

	/**
	 * The cached value of the '{@link #getType() <em>Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getType()
	 * @generated
	 * @ordered
	 */
	protected PrimitiveTypeEnum type = TYPE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.CONSTANT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getInteger() {
		return integer;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setInteger(int newInteger) {
		int oldInteger = integer;
		integer = newInteger;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.CONSTANT__INTEGER, oldInteger, integer));
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
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.CONSTANT__STRING, oldString, string));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public boolean isBoolean() {
		return boolean_;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setBoolean(boolean newBoolean) {
		boolean oldBoolean = boolean_;
		boolean_ = newBoolean;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.CONSTANT__BOOLEAN, oldBoolean, boolean_));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public float getReal() {
		return real;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setReal(float newReal) {
		float oldReal = real;
		real = newReal;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.CONSTANT__REAL, oldReal, real));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PrimitiveTypeEnum getType() {
		return type;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setType(PrimitiveTypeEnum newType) {
		PrimitiveTypeEnum oldType = type;
		type = newType == null ? TYPE_EDEFAULT : newType;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.CONSTANT__TYPE, oldType, type));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PooslPackage.CONSTANT__INTEGER:
				return getInteger();
			case PooslPackage.CONSTANT__STRING:
				return getString();
			case PooslPackage.CONSTANT__BOOLEAN:
				return isBoolean();
			case PooslPackage.CONSTANT__REAL:
				return getReal();
			case PooslPackage.CONSTANT__TYPE:
				return getType();
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
			case PooslPackage.CONSTANT__INTEGER:
				setInteger((Integer)newValue);
				return;
			case PooslPackage.CONSTANT__STRING:
				setString((String)newValue);
				return;
			case PooslPackage.CONSTANT__BOOLEAN:
				setBoolean((Boolean)newValue);
				return;
			case PooslPackage.CONSTANT__REAL:
				setReal((Float)newValue);
				return;
			case PooslPackage.CONSTANT__TYPE:
				setType((PrimitiveTypeEnum)newValue);
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
			case PooslPackage.CONSTANT__INTEGER:
				setInteger(INTEGER_EDEFAULT);
				return;
			case PooslPackage.CONSTANT__STRING:
				setString(STRING_EDEFAULT);
				return;
			case PooslPackage.CONSTANT__BOOLEAN:
				setBoolean(BOOLEAN_EDEFAULT);
				return;
			case PooslPackage.CONSTANT__REAL:
				setReal(REAL_EDEFAULT);
				return;
			case PooslPackage.CONSTANT__TYPE:
				setType(TYPE_EDEFAULT);
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
			case PooslPackage.CONSTANT__INTEGER:
				return integer != INTEGER_EDEFAULT;
			case PooslPackage.CONSTANT__STRING:
				return STRING_EDEFAULT == null ? string != null : !STRING_EDEFAULT.equals(string);
			case PooslPackage.CONSTANT__BOOLEAN:
				return boolean_ != BOOLEAN_EDEFAULT;
			case PooslPackage.CONSTANT__REAL:
				return real != REAL_EDEFAULT;
			case PooslPackage.CONSTANT__TYPE:
				return type != TYPE_EDEFAULT;
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
		result.append(" (integer: ");
		result.append(integer);
		result.append(", string: ");
		result.append(string);
		result.append(", boolean: ");
		result.append(boolean_);
		result.append(", real: ");
		result.append(real);
		result.append(", type: ");
		result.append(type);
		result.append(')');
		return result.toString();
	}

} //ConstantImpl
