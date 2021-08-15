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
 * A representation of the model object '<em><b>Loop Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.LoopExpression#getCondition <em>Condition</em>}</li>
 *   <li>{@link poosl.LoopExpression#getBody <em>Body</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getLoopExpression()
 * @model
 * @generated
 */
public interface LoopExpression extends Expression {
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
	 * @see poosl.PooslPackage#getLoopExpression_Condition()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Expression> getCondition();

	/**
	 * Returns the value of the '<em><b>Body</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Body</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Body</em>' containment reference list.
	 * @see poosl.PooslPackage#getLoopExpression_Body()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Expression> getBody();

} // LoopExpression
