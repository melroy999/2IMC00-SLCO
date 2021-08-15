/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package objects;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Object Lists</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link objects.ObjectLists#getObjectLists <em>Object Lists</em>}</li>
 * </ul>
 * </p>
 *
 * @see objects.ObjectsPackage#getObjectLists()
 * @model
 * @generated
 */
public interface ObjectLists extends EObject {
	/**
	 * Returns the value of the '<em><b>Object Lists</b></em>' containment reference list.
	 * The list contents are of type {@link objects.ObjectList}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Object Lists</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Object Lists</em>' containment reference list.
	 * @see objects.ObjectsPackage#getObjectLists_ObjectLists()
	 * @model containment="true"
	 * @generated
	 */
	EList<ObjectList> getObjectLists();

} // ObjectLists
