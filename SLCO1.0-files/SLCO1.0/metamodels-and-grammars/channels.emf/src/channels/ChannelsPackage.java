/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package channels;

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
 * @see channels.ChannelsFactory
 * @model kind="package"
 * @generated
 */
public interface ChannelsPackage extends EPackage {
	/**
	 * The package name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "channels";

	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "channels";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_PREFIX = "channels";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	ChannelsPackage eINSTANCE = channels.impl.ChannelsPackageImpl.init();

	/**
	 * The meta object id for the '{@link channels.impl.ChannelListImpl <em>Channel List</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see channels.impl.ChannelListImpl
	 * @see channels.impl.ChannelsPackageImpl#getChannelList()
	 * @generated
	 */
	int CHANNEL_LIST = 0;

	/**
	 * The feature id for the '<em><b>Channels</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL_LIST__CHANNELS = 0;

	/**
	 * The number of structural features of the '<em>Channel List</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL_LIST_FEATURE_COUNT = 1;


	/**
	 * The meta object id for the '{@link channels.impl.ChannelListsImpl <em>Channel Lists</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see channels.impl.ChannelListsImpl
	 * @see channels.impl.ChannelsPackageImpl#getChannelLists()
	 * @generated
	 */
	int CHANNEL_LISTS = 1;

	/**
	 * The feature id for the '<em><b>Channel Lists</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL_LISTS__CHANNEL_LISTS = 0;

	/**
	 * The number of structural features of the '<em>Channel Lists</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL_LISTS_FEATURE_COUNT = 1;


	/**
	 * Returns the meta object for class '{@link channels.ChannelList <em>Channel List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Channel List</em>'.
	 * @see channels.ChannelList
	 * @generated
	 */
	EClass getChannelList();

	/**
	 * Returns the meta object for the reference list '{@link channels.ChannelList#getChannels <em>Channels</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Channels</em>'.
	 * @see channels.ChannelList#getChannels()
	 * @see #getChannelList()
	 * @generated
	 */
	EReference getChannelList_Channels();

	/**
	 * Returns the meta object for class '{@link channels.ChannelLists <em>Channel Lists</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Channel Lists</em>'.
	 * @see channels.ChannelLists
	 * @generated
	 */
	EClass getChannelLists();

	/**
	 * Returns the meta object for the containment reference list '{@link channels.ChannelLists#getChannelLists <em>Channel Lists</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Channel Lists</em>'.
	 * @see channels.ChannelLists#getChannelLists()
	 * @see #getChannelLists()
	 * @generated
	 */
	EReference getChannelLists_ChannelLists();

	/**
	 * Returns the factory that creates the instances of the model.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the factory that creates the instances of the model.
	 * @generated
	 */
	ChannelsFactory getChannelsFactory();

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
		 * The meta object literal for the '{@link channels.impl.ChannelListImpl <em>Channel List</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see channels.impl.ChannelListImpl
		 * @see channels.impl.ChannelsPackageImpl#getChannelList()
		 * @generated
		 */
		EClass CHANNEL_LIST = eINSTANCE.getChannelList();

		/**
		 * The meta object literal for the '<em><b>Channels</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CHANNEL_LIST__CHANNELS = eINSTANCE.getChannelList_Channels();

		/**
		 * The meta object literal for the '{@link channels.impl.ChannelListsImpl <em>Channel Lists</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see channels.impl.ChannelListsImpl
		 * @see channels.impl.ChannelsPackageImpl#getChannelLists()
		 * @generated
		 */
		EClass CHANNEL_LISTS = eINSTANCE.getChannelLists();

		/**
		 * The meta object literal for the '<em><b>Channel Lists</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CHANNEL_LISTS__CHANNEL_LISTS = eINSTANCE.getChannelLists_ChannelLists();

	}

} //ChannelsPackage
