/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Atomic</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.Atomic#getStatements <em>Statements</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getAtomic()
 * @model
 * @generated
 */
public interface Atomic extends EObject {
	/**
	 * Returns the value of the '<em><b>Statements</b></em>' reference list.
	 * The list contents are of type {@link poosl.Statement}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Statements</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Statements</em>' reference list.
	 * @see poosl.PooslPackage#getAtomic_Statements()
	 * @model required="true"
	 * @generated
	 */
	EList<Statement> getStatements();

} // Atomic
