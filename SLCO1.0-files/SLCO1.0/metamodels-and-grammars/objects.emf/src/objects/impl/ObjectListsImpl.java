/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package objects.impl;

import java.util.Collection;

import objects.ObjectList;
import objects.ObjectLists;
import objects.ObjectsPackage;

import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Object Lists</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link objects.impl.ObjectListsImpl#getObjectLists <em>Object Lists</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ObjectListsImpl extends EObjectImpl implements ObjectLists {
	/**
	 * The cached value of the '{@link #getObjectLists() <em>Object Lists</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getObjectLists()
	 * @generated
	 * @ordered
	 */
	protected EList<ObjectList> objectLists;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ObjectListsImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ObjectsPackage.Literals.OBJECT_LISTS;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ObjectList> getObjectLists() {
		if (objectLists == null) {
			objectLists = new EObjectContainmentEList<ObjectList>(ObjectList.class, this, ObjectsPackage.OBJECT_LISTS__OBJECT_LISTS);
		}
		return objectLists;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ObjectsPackage.OBJECT_LISTS__OBJECT_LISTS:
				return ((InternalEList<?>)getObjectLists()).basicRemove(otherEnd, msgs);
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
			case ObjectsPackage.OBJECT_LISTS__OBJECT_LISTS:
				return getObjectLists();
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
			case ObjectsPackage.OBJECT_LISTS__OBJECT_LISTS:
				getObjectLists().clear();
				getObjectLists().addAll((Collection<? extends ObjectList>)newValue);
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
			case ObjectsPackage.OBJECT_LISTS__OBJECT_LISTS:
				getObjectLists().clear();
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
			case ObjectsPackage.OBJECT_LISTS__OBJECT_LISTS:
				return objectLists != null && !objectLists.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //ObjectListsImpl
