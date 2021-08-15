/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package objects;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

/**
 * <!-- begin-user-doc -->
 * The <b>Package</b> for the model.
 * It contains accessors for the meta objects to represent
 * <ul>
 *   <li>each class,</li>
 *   <li>each feature of each class,</li>
 *   <li>each enum,</li>
 *   <li>and each data type</li>
 * </ul>
 * <!-- end-user-doc -->
 * @see objects.ObjectsFactory
 * @model kind="package"
 * @generated
 */
public interface ObjectsPackage extends EPackage {
	/**
	 * The package name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "objects";

	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "objects";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_PREFIX = "objects";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	ObjectsPackage eINSTANCE = objects.impl.ObjectsPackageImpl.init();

	/**
	 * The meta object id for the '{@link objects.impl.ObjectListsImpl <em>Object Lists</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see objects.impl.ObjectListsImpl
	 * @see objects.impl.ObjectsPackageImpl#getObjectLists()
	 * @generated
	 */
	int OBJECT_LISTS = 0;

	/**
	 * The feature id for the '<em><b>Object Lists</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OBJECT_LISTS__OBJECT_LISTS = 0;

	/**
	 * The number of structural features of the '<em>Object Lists</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OBJECT_LISTS_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link objects.impl.ObjectListImpl <em>Object List</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see objects.impl.ObjectListImpl
	 * @see objects.impl.ObjectsPackageImpl#getObjectList()
	 * @generated
	 */
	int OBJECT_LIST = 1;

	/**
	 * The feature id for the '<em><b>Objects</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OBJECT_LIST__OBJECTS = 0;

	/**
	 * The number of structural features of the '<em>Object List</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int OBJECT_LIST_FEATURE_COUNT = 1;


	/**
	 * Returns the meta object for class '{@link objects.ObjectLists <em>Object Lists</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Object Lists</em>'.
	 * @see objects.ObjectLists
	 * @generated
	 */
	EClass getObjectLists();

	/**
	 * Returns the meta object for the containment reference list '{@link objects.ObjectLists#getObjectLists <em>Object Lists</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Object Lists</em>'.
	 * @see objects.ObjectLists#getObjectLists()
	 * @see #getObjectLists()
	 * @generated
	 */
	EReference getObjectLists_ObjectLists();

	/**
	 * Returns the meta object for class '{@link objects.ObjectList <em>Object List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Object List</em>'.
	 * @see objects.ObjectList
	 * @generated
	 */
	EClass getObjectList();

	/**
	 * Returns the meta object for the reference list '{@link objects.ObjectList#getObjects <em>Objects</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Objects</em>'.
	 * @see objects.ObjectList#getObjects()
	 * @see #getObjectList()
	 * @generated
	 */
	EReference getObjectList_Objects();

	/**
	 * Returns the factory that creates the instances of the model.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the factory that creates the instances of the model.
	 * @generated
	 */
	ObjectsFactory getObjectsFactory();

	/**
	 * <!-- begin-user-doc -->
	 * Defines literals for the meta objects that represent
	 * <ul>
	 *   <li>each class,</li>
	 *   <li>each feature of each class,</li>
	 *   <li>each enum,</li>
	 *   <li>and each data type</li>
	 * </ul>
	 * <!-- end-user-doc -->
	 * @generated
	 */
	interface Literals {
		/**
		 * The meta object literal for the '{@link objects.impl.ObjectListsImpl <em>Object Lists</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see objects.impl.ObjectListsImpl
		 * @see objects.impl.ObjectsPackageImpl#getObjectLists()
		 * @generated
		 */
		EClass OBJECT_LISTS = eINSTANCE.getObjectLists();

		/**
		 * The meta object literal for the '<em><b>Object Lists</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference OBJECT_LISTS__OBJECT_LISTS = eINSTANCE.getObjectLists_ObjectLists();

		/**
		 * The meta object literal for the '{@link objects.impl.ObjectListImpl <em>Object List</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see objects.impl.ObjectListImpl
		 * @see objects.impl.ObjectsPackageImpl#getObjectList()
		 * @generated
		 */
		EClass OBJECT_LIST = eINSTANCE.getObjectList();

		/**
		 * The meta object literal for the '<em><b>Objects</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference OBJECT_LIST__OBJECTS = eINSTANCE.getObjectList_Objects();

	}

} //ObjectsPackage
