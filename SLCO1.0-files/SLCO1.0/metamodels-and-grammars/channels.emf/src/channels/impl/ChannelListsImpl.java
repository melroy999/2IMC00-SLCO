/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package channels.impl;

import channels.ChannelList;
import channels.ChannelLists;
import channels.ChannelsPackage;

import java.util.Collection;

import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Channel Lists</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link channels.impl.ChannelListsImpl#getChannelLists <em>Channel Lists</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ChannelListsImpl extends EObjectImpl implements ChannelLists {
	/**
	 * The cached value of the '{@link #getChannelLists() <em>Channel Lists</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getChannelLists()
	 * @generated
	 * @ordered
	 */
	protected EList<ChannelList> channelLists;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ChannelListsImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ChannelsPackage.Literals.CHANNEL_LISTS;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ChannelList> getChannelLists() {
		if (channelLists == null) {
			channelLists = new EObjectContainmentEList<ChannelList>(ChannelList.class, this, ChannelsPackage.CHANNEL_LISTS__CHANNEL_LISTS);
		}
		return channelLists;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ChannelsPackage.CHANNEL_LISTS__CHANNEL_LISTS:
				return ((InternalEList<?>)getChannelLists()).basicRemove(otherEnd, msgs);
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
			case ChannelsPackage.CHANNEL_LISTS__CHANNEL_LISTS:
				return getChannelLists();
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
			case ChannelsPackage.CHANNEL_LISTS__CHANNEL_LISTS:
				getChannelLists().clear();
				getChannelLists().addAll((Collection<? extends ChannelList>)newValue);
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
			case ChannelsPackage.CHANNEL_LISTS__CHANNEL_LISTS:
				getChannelLists().clear();
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
			case ChannelsPackage.CHANNEL_LISTS__CHANNEL_LISTS:
				return channelLists != null && !channelLists.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //ChannelListsImpl
