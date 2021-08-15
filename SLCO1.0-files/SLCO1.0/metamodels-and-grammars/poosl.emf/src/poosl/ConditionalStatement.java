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
 * A representation of the model object '<em><b>Conditional Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ConditionalStatement#getCondition <em>Condition</em>}</li>
 *   <li>{@link poosl.ConditionalStatement#getThenClause <em>Then Clause</em>}</li>
 *   <li>{@link poosl.ConditionalStatement#getElseClause <em>Else Clause</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getConditionalStatement()
 * @model
 * @generated
 */
public interface ConditionalStatement extends Statement {
	/**
	 * Returns the value of the '<em><b>Condition</b></em>' reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Condition</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Condition</em>' reference list.
	 * @see poosl.PooslPackage#getConditionalStatement_Condition()
	 * @model required="true"
	 * @generated
	 */
	EList<Expression> getCondition();

	/**
	 * Returns the value of the '<em><b>Then Clause</b></em>' reference list.
	 * The list contents are of type {@link poosl.Statement}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Then Clause</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Then Clause</em>' reference list.
	 * @see poosl.PooslPackage#getConditionalStatement_ThenClause()
	 * @model required="true"
	 * @generated
	 */
	EList<Statement> getThenClause();

	/**
	 * Returns the value of the '<em><b>Else Clause</b></em>' reference list.
	 * The list contents are of type {@link poosl.Statement}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Else Clause</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Else Clause</em>' reference list.
	 * @see poosl.PooslPackage#getConditionalStatement_ElseClause()
	 * @model
	 * @generated
	 */
	EList<Statement> getElseClause();

} // ConditionalStatement
