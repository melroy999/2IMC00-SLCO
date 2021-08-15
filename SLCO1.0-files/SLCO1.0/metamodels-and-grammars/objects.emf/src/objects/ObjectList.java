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
 * A representation of the model object '<em><b>Object List</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link objects.ObjectList#getObjects <em>Objects</em>}</li>
 * </ul>
 * </p>
 *
 * @see objects.ObjectsPackage#getObjectList()
 * @model
 * @generated
 */
public interface ObjectList extends EObject {
	/**
	 * Returns the value of the '<em><b>Objects</b></em>' reference list.
	 * The list contents are of type {@link slco.Object}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Objects</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Objects</em>' reference list.
	 * @see objects.ObjectsPackage#getObjectList_Objects()
	 * @model lower="2"
	 * @generated
	 */
	EList<slco.Object> getObjects();

} // ObjectList
