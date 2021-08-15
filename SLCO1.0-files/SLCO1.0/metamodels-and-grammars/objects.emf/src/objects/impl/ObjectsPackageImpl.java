/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package objects.impl;

import objects.ObjectList;
import objects.ObjectLists;
import objects.ObjectsFactory;
import objects.ObjectsPackage;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

import org.eclipse.emf.ecore.impl.EPackageImpl;

import slco.SlcoPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Package</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class ObjectsPackageImpl extends EPackageImpl implements ObjectsPackage {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass objectListsEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass objectListEClass = null;

	/**
	 * Creates an instance of the model <b>Package</b>, registered with
	 * {@link org.eclipse.emf.ecore.EPackage.Registry EPackage.Registry} by the package
	 * package URI value.
	 * <p>Note: the correct way to create the package is via the static
	 * factory method {@link #init init()}, which also performs
	 * initialization of the package, or returns the registered package,
	 * if one already exists.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see org.eclipse.emf.ecore.EPackage.Registry
	 * @see objects.ObjectsPackage#eNS_URI
	 * @see #init()
	 * @generated
	 */
	private ObjectsPackageImpl() {
		super(eNS_URI, ObjectsFactory.eINSTANCE);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static boolean isInited = false;

	/**
	 * Creates, registers, and initializes the <b>Package</b> for this model, and for any others upon which it depends.
	 * 
	 * <p>This method is used to initialize {@link ObjectsPackage#eINSTANCE} when that field is accessed.
	 * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #eNS_URI
	 * @see #createPackageContents()
	 * @see #initializePackageContents()
	 * @generated
	 */
	public static ObjectsPackage init() {
		if (isInited) return (ObjectsPackage)EPackage.Registry.INSTANCE.getEPackage(ObjectsPackage.eNS_URI);

		// Obtain or create and register package
		ObjectsPackageImpl theObjectsPackage = (ObjectsPackageImpl)(EPackage.Registry.INSTANCE.get(eNS_URI) instanceof ObjectsPackageImpl ? EPackage.Registry.INSTANCE.get(eNS_URI) : new ObjectsPackageImpl());

		isInited = true;

		// Initialize simple dependencies
		SlcoPackage.eINSTANCE.eClass();

		// Create package meta-data objects
		theObjectsPackage.createPackageContents();

		// Initialize created meta-data
		theObjectsPackage.initializePackageContents();

		// Mark meta-data to indicate it can't be changed
		theObjectsPackage.freeze();

  
		// Update the registry and return the package
		EPackage.Registry.INSTANCE.put(ObjectsPackage.eNS_URI, theObjectsPackage);
		return theObjectsPackage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getObjectLists() {
		return objectListsEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getObjectLists_ObjectLists() {
		return (EReference)objectListsEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getObjectList() {
		return objectListEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getObjectList_Objects() {
		return (EReference)objectListEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ObjectsFactory getObjectsFactory() {
		return (ObjectsFactory)getEFactoryInstance();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private boolean isCreated = false;

	/**
	 * Creates the meta-model objects for the package.  This method is
	 * guarded to have no affect on any invocation but its first.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void createPackageContents() {
		if (isCreated) return;
		isCreated = true;

		// Create classes and their features
		objectListsEClass = createEClass(OBJECT_LISTS);
		createEReference(objectListsEClass, OBJECT_LISTS__OBJECT_LISTS);

		objectListEClass = createEClass(OBJECT_LIST);
		createEReference(objectListEClass, OBJECT_LIST__OBJECTS);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private boolean isInitialized = false;

	/**
	 * Complete the initialization of the package and its meta-model.  This
	 * method is guarded to have no affect on any invocation but its first.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void initializePackageContents() {
		if (isInitialized) return;
		isInitialized = true;

		// Initialize package
		setName(eNAME);
		setNsPrefix(eNS_PREFIX);
		setNsURI(eNS_URI);

		// Obtain other dependent packages
		SlcoPackage theSlcoPackage = (SlcoPackage)EPackage.Registry.INSTANCE.getEPackage(SlcoPackage.eNS_URI);

		// Create type parameters

		// Set bounds for type parameters

		// Add supertypes to classes

		// Initialize classes and features; add operations and parameters
		initEClass(objectListsEClass, ObjectLists.class, "ObjectLists", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getObjectLists_ObjectLists(), this.getObjectList(), null, "objectLists", null, 0, -1, ObjectLists.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(objectListEClass, ObjectList.class, "ObjectList", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getObjectList_Objects(), theSlcoPackage.getObject(), null, "objects", null, 2, -1, ObjectList.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		// Create resource
		createResource(eNS_URI);
	}

} //ObjectsPackageImpl
