/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.ConstantExpression;
import nqc.IntegerConstant;
import nqc.NqcPackage;
import nqc.TypeEnum;
import nqc.Variable;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Variable</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.VariableImpl#getName <em>Name</em>}</li>
 *   <li>{@link nqc.impl.VariableImpl#getType <em>Type</em>}</li>
 *   <li>{@link nqc.impl.VariableImpl#getInitialValue <em>Initial Value</em>}</li>
 *   <li>{@link nqc.impl.VariableImpl#getArrayMaxSize <em>Array Max Size</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class VariableImpl extends EObjectImpl implements Variable {
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
	 * The default value of the '{@link #getType() <em>Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getType()
	 * @generated
	 * @ordered
	 */
	protected static final TypeEnum TYPE_EDEFAULT = TypeEnum.INTEGER;

	/**
	 * The cached value of the '{@link #getType() <em>Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getType()
	 * @generated
	 * @ordered
	 */
	protected TypeEnum type = TYPE_EDEFAULT;

	/**
	 * The cached value of the '{@link #getInitialValue() <em>Initial Value</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInitialValue()
	 * @generated
	 * @ordered
	 */
	protected ConstantExpression initialValue;

	/**
	 * The cached value of the '{@link #getArrayMaxSize() <em>Array Max Size</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getArrayMaxSize()
	 * @generated
	 * @ordered
	 */
	protected IntegerConstant arrayMaxSize;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected VariableImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getVariable();
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
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.VARIABLE__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TypeEnum getType() {
		return type;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setType(TypeEnum newType) {
		TypeEnum oldType = type;
		type = newType == null ? TYPE_EDEFAULT : newType;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.VARIABLE__TYPE, oldType, type));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ConstantExpression getInitialValue() {
		return initialValue;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetInitialValue(ConstantExpression newInitialValue, NotificationChain msgs) {
		ConstantExpression oldInitialValue = initialValue;
		initialValue = newInitialValue;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.VARIABLE__INITIAL_VALUE, oldInitialValue, newInitialValue);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setInitialValue(ConstantExpression newInitialValue) {
		if (newInitialValue != initialValue) {
			NotificationChain msgs = null;
			if (initialValue != null)
				msgs = ((InternalEObject)initialValue).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.VARIABLE__INITIAL_VALUE, null, msgs);
			if (newInitialValue != null)
				msgs = ((InternalEObject)newInitialValue).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.VARIABLE__INITIAL_VALUE, null, msgs);
			msgs = basicSetInitialValue(newInitialValue, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.VARIABLE__INITIAL_VALUE, newInitialValue, newInitialValue));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IntegerConstant getArrayMaxSize() {
		if (arrayMaxSize != null && arrayMaxSize.eIsProxy()) {
			InternalEObject oldArrayMaxSize = (InternalEObject)arrayMaxSize;
			arrayMaxSize = (IntegerConstant)eResolveProxy(oldArrayMaxSize);
			if (arrayMaxSize != oldArrayMaxSize) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, NqcPackage.VARIABLE__ARRAY_MAX_SIZE, oldArrayMaxSize, arrayMaxSize));
			}
		}
		return arrayMaxSize;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IntegerConstant basicGetArrayMaxSize() {
		return arrayMaxSize;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setArrayMaxSize(IntegerConstant newArrayMaxSize) {
		IntegerConstant oldArrayMaxSize = arrayMaxSize;
		arrayMaxSize = newArrayMaxSize;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.VARIABLE__ARRAY_MAX_SIZE, oldArrayMaxSize, arrayMaxSize));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.VARIABLE__INITIAL_VALUE:
				return basicSetInitialValue(null, msgs);
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
			case NqcPackage.VARIABLE__NAME:
				return getName();
			case NqcPackage.VARIABLE__TYPE:
				return getType();
			case NqcPackage.VARIABLE__INITIAL_VALUE:
				return getInitialValue();
			case NqcPackage.VARIABLE__ARRAY_MAX_SIZE:
				if (resolve) return getArrayMaxSize();
				return basicGetArrayMaxSize();
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
			case NqcPackage.VARIABLE__NAME:
				setName((String)newValue);
				return;
			case NqcPackage.VARIABLE__TYPE:
				setType((TypeEnum)newValue);
				return;
			case NqcPackage.VARIABLE__INITIAL_VALUE:
				setInitialValue((ConstantExpression)newValue);
				return;
			case NqcPackage.VARIABLE__ARRAY_MAX_SIZE:
				setArrayMaxSize((IntegerConstant)newValue);
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
			case NqcPackage.VARIABLE__NAME:
				setName(NAME_EDEFAULT);
				return;
			case NqcPackage.VARIABLE__TYPE:
				setType(TYPE_EDEFAULT);
				return;
			case NqcPackage.VARIABLE__INITIAL_VALUE:
				setInitialValue((ConstantExpression)null);
				return;
			case NqcPackage.VARIABLE__ARRAY_MAX_SIZE:
				setArrayMaxSize((IntegerConstant)null);
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
			case NqcPackage.VARIABLE__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case NqcPackage.VARIABLE__TYPE:
				return type != TYPE_EDEFAULT;
			case NqcPackage.VARIABLE__INITIAL_VALUE:
				return initialValue != null;
			case NqcPackage.VARIABLE__ARRAY_MAX_SIZE:
				return arrayMaxSize != null;
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
		result.append(" (Name: ");
		result.append(name);
		result.append(", Type: ");
		result.append(type);
		result.append(')');
		return result.toString();
	}

} //VariableImpl
