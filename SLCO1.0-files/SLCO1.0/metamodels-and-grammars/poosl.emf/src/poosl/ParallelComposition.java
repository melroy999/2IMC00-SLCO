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
 * A representation of the model object '<em><b>Parallel Composition</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ParallelComposition#getStatementLists <em>Statement Lists</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getParallelComposition()
 * @model
 * @generated
 */
public interface ParallelComposition extends Statement {
	/**
	 * Returns the value of the '<em><b>Statement Lists</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.StatementList}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Statement Lists</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Statement Lists</em>' containment reference list.
	 * @see poosl.PooslPackage#getParallelComposition_StatementLists()
	 * @model containment="true" lower="2"
	 * @generated
	 */
	EList<StatementList> getStatementLists();

} // ParallelComposition
