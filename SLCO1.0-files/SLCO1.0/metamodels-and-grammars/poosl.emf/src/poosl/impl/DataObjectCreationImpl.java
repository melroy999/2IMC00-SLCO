/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import poosl.DataClass;
import poosl.DataObjectCreation;
import poosl.PooslPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Data Object Creation</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.DataObjectCreationImpl#getDataClass <em>Data Class</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class DataObjectCreationImpl extends ExpressionImpl implements DataObjectCreation {
	/**
	 * The cached value of the '{@link #getDataClass() <em>Data Class</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDataClass()
	 * @generated
	 * @ordered
	 */
	protected DataClass dataClass;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected DataObjectCreationImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.DATA_OBJECT_CREATION;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DataClass getDataClass() {
		if (dataClass != null && dataClass.eIsProxy()) {
			InternalEObject oldDataClass = (InternalEObject)dataClass;
			dataClass = (DataClass)eResolveProxy(oldDataClass);
			if (dataClass != oldDataClass) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.DATA_OBJECT_CREATION__DATA_CLASS, oldDataClass, dataClass));
			}
		}
		return dataClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DataClass basicGetDataClass() {
		return dataClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setDataClass(DataClass newDataClass) {
		DataClass oldDataClass = dataClass;
		dataClass = newDataClass;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.DATA_OBJECT_CREATION__DATA_CLASS, oldDataClass, dataClass));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PooslPackage.DATA_OBJECT_CREATION__DATA_CLASS:
				if (resolve) return getDataClass();
				return basicGetDataClass();
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
			case PooslPackage.DATA_OBJECT_CREATION__DATA_CLASS:
				setDataClass((DataClass)newValue);
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
			case PooslPackage.DATA_OBJECT_CREATION__DATA_CLASS:
				setDataClass((DataClass)null);
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
			case PooslPackage.DATA_OBJECT_CREATION__DATA_CLASS:
				return dataClass != null;
		}
		return super.eIsSet(featureID);
	}

} //DataObjectCreationImpl
