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
 * A representation of the model object '<em><b>Conditional Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ConditionalExpression#getCondition <em>Condition</em>}</li>
 *   <li>{@link poosl.ConditionalExpression#getThenClause <em>Then Clause</em>}</li>
 *   <li>{@link poosl.ConditionalExpression#getElseClause <em>Else Clause</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getConditionalExpression()
 * @model
 * @generated
 */
public interface ConditionalExpression extends Expression {
	/**
	 * Returns the value of the '<em><b>Condition</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Condition</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Condition</em>' containment reference list.
	 * @see poosl.PooslPackage#getConditionalExpression_Condition()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Expression> getCondition();

	/**
	 * Returns the value of the '<em><b>Then Clause</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Then Clause</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Then Clause</em>' containment reference list.
	 * @see poosl.PooslPackage#getConditionalExpression_ThenClause()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Expression> getThenClause();

	/**
	 * Returns the value of the '<em><b>Else Clause</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Else Clause</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Else Clause</em>' containment reference list.
	 * @see poosl.PooslPackage#getConditionalExpression_ElseClause()
	 * @model containment="true"
	 * @generated
	 */
	EList<Expression> getElseClause();

} // ConditionalExpression
