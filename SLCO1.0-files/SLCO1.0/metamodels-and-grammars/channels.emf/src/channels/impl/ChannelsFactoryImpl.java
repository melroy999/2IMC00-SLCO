/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package channels.impl;

import channels.*;

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
public class ChannelsFactoryImpl extends EFactoryImpl implements ChannelsFactory {
	/**
	 * Creates the default factory implementation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static ChannelsFactory init() {
		try {
			ChannelsFactory theChannelsFactory = (ChannelsFactory)EPackage.Registry.INSTANCE.getEFactory("channels"); 
			if (theChannelsFactory != null) {
				return theChannelsFactory;
			}
		}
		catch (Exception exception) {
			EcorePlugin.INSTANCE.log(exception);
		}
		return new ChannelsFactoryImpl();
	}

	/**
	 * Creates an instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ChannelsFactoryImpl() {
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
			case ChannelsPackage.CHANNEL_LIST: return createChannelList();
			case ChannelsPackage.CHANNEL_LISTS: return createChannelLists();
			default:
				throw new IllegalArgumentException("The class '" + eClass.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ChannelList createChannelList() {
		ChannelListImpl channelList = new ChannelListImpl();
		return channelList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ChannelLists createChannelLists() {
		ChannelListsImpl channelLists = new ChannelListsImpl();
		return channelLists;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ChannelsPackage getChannelsPackage() {
		return (ChannelsPackage)getEPackage();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @deprecated
	 * @generated
	 */
	@Deprecated
	public static ChannelsPackage getPackage() {
		return ChannelsPackage.eINSTANCE;
	}

} //ChannelsFactoryImpl
