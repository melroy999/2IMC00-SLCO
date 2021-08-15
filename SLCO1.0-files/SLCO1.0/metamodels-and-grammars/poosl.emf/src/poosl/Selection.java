/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Selection</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.Selection#getStatementLists <em>Statement Lists</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getSelection()
 * @model
 * @generated
 */
public interface Selection extends Statement {
	/**
	 * Returns the value of the '<em><b>Statement Lists</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.StatementList}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Statement Lists</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Statement Lists</em>' containment reference list.
	 * @see poosl.PooslPackage#getSelection_StatementLists()
	 * @model containment="true" lower="2"
	 * @generated
	 */
	EList<StatementList> getStatementLists();

} // Selection
