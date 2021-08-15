/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package channels;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @see channels.ChannelsPackage
 * @generated
 */
public interface ChannelsFactory extends EFactory {
	/**
	 * The singleton instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	ChannelsFactory eINSTANCE = channels.impl.ChannelsFactoryImpl.init();

	/**
	 * Returns a new object of class '<em>Channel List</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Channel List</em>'.
	 * @generated
	 */
	ChannelList createChannelList();

	/**
	 * Returns a new object of class '<em>Channel Lists</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Channel Lists</em>'.
	 * @generated
	 */
	ChannelLists createChannelLists();

	/**
	 * Returns the package supported by this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package supported by this factory.
	 * @generated
	 */
	ChannelsPackage getChannelsPackage();

} //ChannelsFactory
