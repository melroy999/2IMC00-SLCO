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
 * A representation of the model object '<em><b>Expression List</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ExpressionList#getExpressions <em>Expressions</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getExpressionList()
 * @model
 * @generated
 */
public interface ExpressionList extends EObject {
	/**
	 * Returns the value of the '<em><b>Expressions</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Expressions</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Expressions</em>' containment reference list.
	 * @see poosl.PooslPackage#getExpressionList_Expressions()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Expression> getExpressions();

} // ExpressionList
