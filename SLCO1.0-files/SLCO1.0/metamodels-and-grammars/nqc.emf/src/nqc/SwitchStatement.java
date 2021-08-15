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
 * A representation of the model object '<em><b>Switch Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SwitchStatement#getCondition <em>Condition</em>}</li>
 *   <li>{@link nqc.SwitchStatement#getCases <em>Cases</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSwitchStatement()
 * @model
 * @generated
 */
public interface SwitchStatement extends ControlStructure {
	/**
	 * Returns the value of the '<em><b>Condition</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Condition</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Condition</em>' reference.
	 * @see #setCondition(Expression)
	 * @see nqc.NqcPackage#getSwitchStatement_Condition()
	 * @model required="true" ordered="false"
	 * @generated
	 */
	Expression getCondition();

	/**
	 * Sets the value of the '{@link nqc.SwitchStatement#getCondition <em>Condition</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Condition</em>' reference.
	 * @see #getCondition()
	 * @generated
	 */
	void setCondition(Expression value);

	/**
	 * Returns the value of the '<em><b>Cases</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.Case}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Cases</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Cases</em>' containment reference list.
	 * @see nqc.NqcPackage#getSwitchStatement_Cases()
	 * @model containment="true"
	 * @generated
	 */
	EList<Case> getCases();

} // SwitchStatement
