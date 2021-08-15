/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Class List</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.ClassList#getClasses <em>Classes</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getClassList()
 * @model
 * @generated
 */
public interface ClassList extends EObject {
	/**
	 * Returns the value of the '<em><b>Classes</b></em>' reference list.
	 * The list contents are of type {@link slco.Class}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Classes</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Classes</em>' reference list.
	 * @see slco2nqc.Slco2nqcPackage#getClassList_Classes()
	 * @model
	 * @generated
	 */
	EList<slco.Class> getClasses();

} // ClassList
