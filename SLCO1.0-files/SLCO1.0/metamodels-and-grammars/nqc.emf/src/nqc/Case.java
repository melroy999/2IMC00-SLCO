/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Case</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.Case#isIsDefault <em>Is Default</em>}</li>
 *   <li>{@link nqc.Case#getLabel <em>Label</em>}</li>
 *   <li>{@link nqc.Case#getStatements <em>Statements</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getCase()
 * @model
 * @generated
 */
public interface Case extends EObject {
	/**
	 * Returns the value of the '<em><b>Is Default</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Is Default</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Is Default</em>' attribute.
	 * @see #setIsDefault(boolean)
	 * @see nqc.NqcPackage#getCase_IsDefault()
	 * @model unique="false" required="true" ordered="false"
	 * @generated
	 */
	boolean isIsDefault();

	/**
	 * Sets the value of the '{@link nqc.Case#isIsDefault <em>Is Default</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Is Default</em>' attribute.
	 * @see #isIsDefault()
	 * @generated
	 */
	void setIsDefault(boolean value);

	/**
	 * Returns the value of the '<em><b>Label</b></em>' reference list.
	 * The list contents are of type {@link nqc.ConstantExpression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Label</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Label</em>' reference list.
	 * @see nqc.NqcPackage#getCase_Label()
	 * @model required="true" ordered="false"
	 * @generated
	 */
	EList<ConstantExpression> getLabel();

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
	 * @see nqc.NqcPackage#getCase_Statements()
	 * @model containment="true"
	 * @generated
	 */
	EList<Statement> getStatements();

} // Case
