/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package channels;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

import slco.Channel;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Channel List</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link channels.ChannelList#getChannels <em>Channels</em>}</li>
 * </ul>
 * </p>
 *
 * @see channels.ChannelsPackage#getChannelList()
 * @model
 * @generated
 */
public interface ChannelList extends EObject {
	/**
	 * Returns the value of the '<em><b>Channels</b></em>' reference list.
	 * The list contents are of type {@link slco.Channel}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Channels</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Channels</em>' reference list.
	 * @see channels.ChannelsPackage#getChannelList_Channels()
	 * @model
	 * @generated
	 */
	EList<Channel> getChannels();

} // ChannelList
