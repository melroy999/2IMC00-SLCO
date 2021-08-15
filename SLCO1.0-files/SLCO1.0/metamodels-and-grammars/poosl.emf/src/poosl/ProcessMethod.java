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
 * A representation of the model object '<em><b>Process Method</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ProcessMethod#getInputParameters <em>Input Parameters</em>}</li>
 *   <li>{@link poosl.ProcessMethod#getOutputParameters <em>Output Parameters</em>}</li>
 *   <li>{@link poosl.ProcessMethod#getVariables <em>Variables</em>}</li>
 *   <li>{@link poosl.ProcessMethod#getName <em>Name</em>}</li>
 *   <li>{@link poosl.ProcessMethod#getBody <em>Body</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getProcessMethod()
 * @model
 * @generated
 */
public interface ProcessMethod extends EObject {
	/**
	 * Returns the value of the '<em><b>Input Parameters</b></em>' reference list.
	 * The list contents are of type {@link poosl.Parameter}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Input Parameters</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Input Parameters</em>' reference list.
	 * @see poosl.PooslPackage#getProcessMethod_InputParameters()
	 * @model
	 * @generated
	 */
	EList<Parameter> getInputParameters();

	/**
	 * Returns the value of the '<em><b>Output Parameters</b></em>' reference list.
	 * The list contents are of type {@link poosl.Parameter}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Output Parameters</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Output Parameters</em>' reference list.
	 * @see poosl.PooslPackage#getProcessMethod_OutputParameters()
	 * @model
	 * @generated
	 */
	EList<Parameter> getOutputParameters();

	/**
	 * Returns the value of the '<em><b>Variables</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Variable}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Variables</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Variables</em>' containment reference list.
	 * @see poosl.PooslPackage#getProcessMethod_Variables()
	 * @model containment="true"
	 * @generated
	 */
	EList<Variable> getVariables();

	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see poosl.PooslPackage#getProcessMethod_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link poosl.ProcessMethod#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Body</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Statement}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Body</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Body</em>' containment reference list.
	 * @see poosl.PooslPackage#getProcessMethod_Body()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Statement> getBody();

} // ProcessMethod
