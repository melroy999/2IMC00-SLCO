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
 * A representation of the model object '<em><b>Abort</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.Abort#getBody <em>Body</em>}</li>
 *   <li>{@link poosl.Abort#getCondition <em>Condition</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getAbort()
 * @model
 * @generated
 */
public interface Abort extends Statement {
	/**
	 * Returns the value of the '<em><b>Body</b></em>' reference list.
	 * The list contents are of type {@link poosl.Statement}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Body</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Body</em>' reference list.
	 * @see poosl.PooslPackage#getAbort_Body()
	 * @model required="true"
	 * @generated
	 */
	EList<Statement> getBody();

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
	 * @see poosl.PooslPackage#getAbort_Condition()
	 * @model required="true"
	 * @generated
	 */
	EList<Expression> getCondition();

} // Abort
