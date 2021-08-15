/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package channels.impl;

import channels.ChannelList;
import channels.ChannelLists;
import channels.ChannelsFactory;
import channels.ChannelsPackage;

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
public class ChannelsPackageImpl extends EPackageImpl implements ChannelsPackage {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass channelListEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass channelListsEClass = null;

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
	 * @see channels.ChannelsPackage#eNS_URI
	 * @see #init()
	 * @generated
	 */
	private ChannelsPackageImpl() {
		super(eNS_URI, ChannelsFactory.eINSTANCE);
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
	 * <p>This method is used to initialize {@link ChannelsPackage#eINSTANCE} when that field is accessed.
	 * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #eNS_URI
	 * @see #createPackageContents()
	 * @see #initializePackageContents()
	 * @generated
	 */
	public static ChannelsPackage init() {
		if (isInited) return (ChannelsPackage)EPackage.Registry.INSTANCE.getEPackage(ChannelsPackage.eNS_URI);

		// Obtain or create and register package
		ChannelsPackageImpl theChannelsPackage = (ChannelsPackageImpl)(EPackage.Registry.INSTANCE.get(eNS_URI) instanceof ChannelsPackageImpl ? EPackage.Registry.INSTANCE.get(eNS_URI) : new ChannelsPackageImpl());

		isInited = true;

		// Initialize simple dependencies
		SlcoPackage.eINSTANCE.eClass();

		// Create package meta-data objects
		theChannelsPackage.createPackageContents();

		// Initialize created meta-data
		theChannelsPackage.initializePackageContents();

		// Mark meta-data to indicate it can't be changed
		theChannelsPackage.freeze();

  
		// Update the registry and return the package
		EPackage.Registry.INSTANCE.put(ChannelsPackage.eNS_URI, theChannelsPackage);
		return theChannelsPackage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getChannelList() {
		return channelListEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getChannelList_Channels() {
		return (EReference)channelListEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getChannelLists() {
		return channelListsEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getChannelLists_ChannelLists() {
		return (EReference)channelListsEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ChannelsFactory getChannelsFactory() {
		return (ChannelsFactory)getEFactoryInstance();
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
		channelListEClass = createEClass(CHANNEL_LIST);
		createEReference(channelListEClass, CHANNEL_LIST__CHANNELS);

		channelListsEClass = createEClass(CHANNEL_LISTS);
		createEReference(channelListsEClass, CHANNEL_LISTS__CHANNEL_LISTS);
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
		initEClass(channelListEClass, ChannelList.class, "ChannelList", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getChannelList_Channels(), theSlcoPackage.getChannel(), null, "channels", null, 0, -1, ChannelList.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(channelListsEClass, ChannelLists.class, "ChannelLists", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getChannelLists_ChannelLists(), this.getChannelList(), null, "channelLists", null, 0, -1, ChannelLists.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		// Create resource
		createResource(eNS_URI);
	}

} //ChannelsPackageImpl
