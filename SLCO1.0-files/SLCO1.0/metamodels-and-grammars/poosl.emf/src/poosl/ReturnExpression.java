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
 * A representation of the model object '<em><b>Return Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ReturnExpression#getReturnExpressions <em>Return Expressions</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getReturnExpression()
 * @model
 * @generated
 */
public interface ReturnExpression extends Expression {
	/**
	 * Returns the value of the '<em><b>Return Expressions</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Return Expressions</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Return Expressions</em>' containment reference list.
	 * @see poosl.PooslPackage#getReturnExpression_ReturnExpressions()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Expression> getReturnExpressions();

} // ReturnExpression
