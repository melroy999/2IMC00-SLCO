/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package objects.impl;

import objects.*;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.impl.EFactoryImpl;

import org.eclipse.emf.ecore.plugin.EcorePlugin;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Factory</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class ObjectsFactoryImpl extends EFactoryImpl implements ObjectsFactory {
	/**
	 * Creates the default factory implementation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static ObjectsFactory init() {
		try {
			ObjectsFactory theObjectsFactory = (ObjectsFactory)EPackage.Registry.INSTANCE.getEFactory("objects"); 
			if (theObjectsFactory != null) {
				return theObjectsFactory;
			}
		}
		catch (Exception exception) {
			EcorePlugin.INSTANCE.log(exception);
		}
		return new ObjectsFactoryImpl();
	}

	/**
	 * Creates an instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ObjectsFactoryImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EObject create(EClass eClass) {
		switch (eClass.getClassifierID()) {
			case ObjectsPackage.OBJECT_LISTS: return createObjectLists();
			case ObjectsPackage.OBJECT_LIST: return createObjectList();
			default:
				throw new IllegalArgumentException("The class '" + eClass.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ObjectLists createObjectLists() {
		ObjectListsImpl objectLists = new ObjectListsImpl();
		return objectLists;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ObjectList createObjectList() {
		ObjectListImpl objectList = new ObjectListImpl();
		return objectList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ObjectsPackage getObjectsPackage() {
		return (ObjectsPackage)getEPackage();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @deprecated
	 * @generated
	 */
	@Deprecated
	public static ObjectsPackage getPackage() {
		return ObjectsPackage.eINSTANCE;
	}

} //ObjectsFactoryImpl
