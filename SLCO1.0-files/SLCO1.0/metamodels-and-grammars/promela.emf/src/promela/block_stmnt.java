/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>block stmnt</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.block_stmnt#getSequence <em>Sequence</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getblock_stmnt()
 * @model
 * @generated
 */
public interface block_stmnt extends stmnt {
	/**
	 * Returns the value of the '<em><b>Sequence</b></em>' containment reference list.
	 * The list contents are of type {@link promela.step}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sequence</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sequence</em>' containment reference list.
	 * @see promela.PromelaPackage#getblock_stmnt_Sequence()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<step> getSequence();

} // block_stmnt
