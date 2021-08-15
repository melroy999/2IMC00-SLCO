/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package objects.impl;

import java.util.Collection;

import objects.ObjectList;
import objects.ObjectsPackage;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectResolvingEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Object List</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link objects.impl.ObjectListImpl#getObjects <em>Objects</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ObjectListImpl extends EObjectImpl implements ObjectList {
	/**
	 * The cached value of the '{@link #getObjects() <em>Objects</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getObjects()
	 * @generated
	 * @ordered
	 */
	protected EList<slco.Object> objects;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ObjectListImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ObjectsPackage.Literals.OBJECT_LIST;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<slco.Object> getObjects() {
		if (objects == null) {
			objects = new EObjectResolvingEList<slco.Object>(slco.Object.class, this, ObjectsPackage.OBJECT_LIST__OBJECTS);
		}
		return objects;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case ObjectsPackage.OBJECT_LIST__OBJECTS:
				return getObjects();
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
			case ObjectsPackage.OBJECT_LIST__OBJECTS:
				getObjects().clear();
				getObjects().addAll((Collection<? extends slco.Object>)newValue);
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
			case ObjectsPackage.OBJECT_LIST__OBJECTS:
				getObjects().clear();
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
			case ObjectsPackage.OBJECT_LIST__OBJECTS:
				return objects != null && !objects.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //ObjectListImpl
