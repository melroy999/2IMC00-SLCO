/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Programs</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.Programs#getPrograms <em>Programs</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getPrograms()
 * @model
 * @generated
 */
public interface Programs extends EObject {
	/**
	 * Returns the value of the '<em><b>Programs</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.Program}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Programs</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Programs</em>' containment reference list.
	 * @see nqc.NqcPackage#getPrograms_Programs()
	 * @model containment="true"
	 * @generated
	 */
	EList<Program> getPrograms();

} // Programs
