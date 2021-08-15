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
 * A representation of the model object '<em><b>Function Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.FunctionCall#getCallee <em>Callee</em>}</li>
 *   <li>{@link nqc.FunctionCall#getParameters <em>Parameters</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getFunctionCall()
 * @model
 * @generated
 */
public interface FunctionCall extends CallStatement {
	/**
	 * Returns the value of the '<em><b>Callee</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Callee</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Callee</em>' reference.
	 * @see #setCallee(Function)
	 * @see nqc.NqcPackage#getFunctionCall_Callee()
	 * @model required="true" ordered="false"
	 * @generated
	 */
	Function getCallee();

	/**
	 * Sets the value of the '{@link nqc.FunctionCall#getCallee <em>Callee</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Callee</em>' reference.
	 * @see #getCallee()
	 * @generated
	 */
	void setCallee(Function value);

	/**
	 * Returns the value of the '<em><b>Parameters</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameters</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameters</em>' containment reference list.
	 * @see nqc.NqcPackage#getFunctionCall_Parameters()
	 * @model containment="true"
	 * @generated
	 */
	EList<Expression> getParameters();

} // FunctionCall
