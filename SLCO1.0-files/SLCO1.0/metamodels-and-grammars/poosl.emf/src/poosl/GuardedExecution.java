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
 * A representation of the model object '<em><b>Guarded Execution</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.GuardedExecution#getStatement <em>Statement</em>}</li>
 *   <li>{@link poosl.GuardedExecution#getGuard <em>Guard</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getGuardedExecution()
 * @model
 * @generated
 */
public interface GuardedExecution extends Statement {
	/**
	 * Returns the value of the '<em><b>Statement</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Statement</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Statement</em>' containment reference.
	 * @see #setStatement(Statement)
	 * @see poosl.PooslPackage#getGuardedExecution_Statement()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Statement getStatement();

	/**
	 * Sets the value of the '{@link poosl.GuardedExecution#getStatement <em>Statement</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Statement</em>' containment reference.
	 * @see #getStatement()
	 * @generated
	 */
	void setStatement(Statement value);

	/**
	 * Returns the value of the '<em><b>Guard</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Guard</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Guard</em>' containment reference list.
	 * @see poosl.PooslPackage#getGuardedExecution_Guard()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Expression> getGuard();

} // GuardedExecution
