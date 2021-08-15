/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package channels.impl;

import channels.ChannelList;
import channels.ChannelsPackage;

import java.util.Collection;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectResolvingEList;

import slco.Channel;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Channel List</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link channels.impl.ChannelListImpl#getChannels <em>Channels</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ChannelListImpl extends EObjectImpl implements ChannelList {
	/**
	 * The cached value of the '{@link #getChannels() <em>Channels</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getChannels()
	 * @generated
	 * @ordered
	 */
	protected EList<Channel> channels;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ChannelListImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ChannelsPackage.Literals.CHANNEL_LIST;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Channel> getChannels() {
		if (channels == null) {
			channels = new EObjectResolvingEList<Channel>(Channel.class, this, ChannelsPackage.CHANNEL_LIST__CHANNELS);
		}
		return channels;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case ChannelsPackage.CHANNEL_LIST__CHANNELS:
				return getChannels();
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
			case ChannelsPackage.CHANNEL_LIST__CHANNELS:
				getChannels().clear();
				getChannels().addAll((Collection<? extends Channel>)newValue);
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
			case ChannelsPackage.CHANNEL_LIST__CHANNELS:
				getChannels().clear();
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
			case ChannelsPackage.CHANNEL_LIST__CHANNELS:
				return channels != null && !channels.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //ChannelListImpl
