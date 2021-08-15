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
 * A representation of the model object '<em><b>Process Method Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ProcessMethodCall#getMethod <em>Method</em>}</li>
 *   <li>{@link poosl.ProcessMethodCall#getParameterExpressions <em>Parameter Expressions</em>}</li>
 *   <li>{@link poosl.ProcessMethodCall#getVariables <em>Variables</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getProcessMethodCall()
 * @model
 * @generated
 */
public interface ProcessMethodCall extends Statement {
	/**
	 * Returns the value of the '<em><b>Method</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Method</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Method</em>' reference.
	 * @see #setMethod(ProcessMethod)
	 * @see poosl.PooslPackage#getProcessMethodCall_Method()
	 * @model required="true"
	 * @generated
	 */
	ProcessMethod getMethod();

	/**
	 * Sets the value of the '{@link poosl.ProcessMethodCall#getMethod <em>Method</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Method</em>' reference.
	 * @see #getMethod()
	 * @generated
	 */
	void setMethod(ProcessMethod value);

	/**
	 * Returns the value of the '<em><b>Parameter Expressions</b></em>' reference list.
	 * The list contents are of type {@link poosl.ExpressionList}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter Expressions</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter Expressions</em>' reference list.
	 * @see poosl.PooslPackage#getProcessMethodCall_ParameterExpressions()
	 * @model
	 * @generated
	 */
	EList<ExpressionList> getParameterExpressions();

	/**
	 * Returns the value of the '<em><b>Variables</b></em>' reference list.
	 * The list contents are of type {@link poosl.Variable}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Variables</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Variables</em>' reference list.
	 * @see poosl.PooslPackage#getProcessMethodCall_Variables()
	 * @model
	 * @generated
	 */
	EList<Variable> getVariables();

} // ProcessMethodCall
