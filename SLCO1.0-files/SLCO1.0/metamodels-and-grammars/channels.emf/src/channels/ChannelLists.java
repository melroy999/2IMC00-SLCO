/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package channels;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Channel Lists</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link channels.ChannelLists#getChannelLists <em>Channel Lists</em>}</li>
 * </ul>
 * </p>
 *
 * @see channels.ChannelsPackage#getChannelLists()
 * @model
 * @generated
 */
public interface ChannelLists extends EObject {
	/**
	 * Returns the value of the '<em><b>Channel Lists</b></em>' containment reference list.
	 * The list contents are of type {@link channels.ChannelList}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Channel Lists</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Channel Lists</em>' containment reference list.
	 * @see channels.ChannelsPackage#getChannelLists_ChannelLists()
	 * @model containment="true"
	 * @generated
	 */
	EList<ChannelList> getChannelLists();

} // ChannelLists
