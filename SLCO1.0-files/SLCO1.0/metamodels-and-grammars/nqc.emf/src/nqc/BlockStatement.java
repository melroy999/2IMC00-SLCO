/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Block Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.BlockStatement#getLocalVariables <em>Local Variables</em>}</li>
 *   <li>{@link nqc.BlockStatement#getStatements <em>Statements</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getBlockStatement()
 * @model
 * @generated
 */
public interface BlockStatement extends Statement {
	/**
	 * Returns the value of the '<em><b>Local Variables</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.Variable}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Local Variables</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Local Variables</em>' containment reference list.
	 * @see nqc.NqcPackage#getBlockStatement_LocalVariables()
	 * @model containment="true" ordered="false"
	 * @generated
	 */
	EList<Variable> getLocalVariables();

	/**
	 * Returns the value of the '<em><b>Statements</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.Statement}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Statements</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Statements</em>' containment reference list.
	 * @see nqc.NqcPackage#getBlockStatement_Statements()
	 * @model containment="true"
	 * @generated
	 */
	EList<Statement> getStatements();

} // BlockStatement
