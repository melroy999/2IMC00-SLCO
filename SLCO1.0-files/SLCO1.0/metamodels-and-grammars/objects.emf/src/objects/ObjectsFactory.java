/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package objects;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @see objects.ObjectsPackage
 * @generated
 */
public interface ObjectsFactory extends EFactory {
	/**
	 * The singleton instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	ObjectsFactory eINSTANCE = objects.impl.ObjectsFactoryImpl.init();

	/**
	 * Returns a new object of class '<em>Object Lists</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Object Lists</em>'.
	 * @generated
	 */
	ObjectLists createObjectLists();

	/**
	 * Returns a new object of class '<em>Object List</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Object List</em>'.
	 * @generated
	 */
	ObjectList createObjectList();

	/**
	 * Returns the package supported by this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package supported by this factory.
	 * @generated
	 */
	ObjectsPackage getObjectsPackage();

} //ObjectsFactory
