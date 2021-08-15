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
 * A representation of the model object '<em><b>Process Class</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ProcessClass#getVariables <em>Variables</em>}</li>
 *   <li>{@link poosl.ProcessClass#getMethods <em>Methods</em>}</li>
 *   <li>{@link poosl.ProcessClass#getSuperClass <em>Super Class</em>}</li>
 *   <li>{@link poosl.ProcessClass#getParameters <em>Parameters</em>}</li>
 *   <li>{@link poosl.ProcessClass#getPorts <em>Ports</em>}</li>
 *   <li>{@link poosl.ProcessClass#getInitialMethodCall <em>Initial Method Call</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getProcessClass()
 * @model
 * @generated
 */
public interface ProcessClass extends poosl.Class {
	/**
	 * Returns the value of the '<em><b>Variables</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Variable}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Variables</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Variables</em>' containment reference list.
	 * @see poosl.PooslPackage#getProcessClass_Variables()
	 * @model containment="true"
	 * @generated
	 */
	EList<Variable> getVariables();

	/**
	 * Returns the value of the '<em><b>Methods</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.ProcessMethod}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Methods</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Methods</em>' containment reference list.
	 * @see poosl.PooslPackage#getProcessClass_Methods()
	 * @model containment="true"
	 * @generated
	 */
	EList<ProcessMethod> getMethods();

	/**
	 * Returns the value of the '<em><b>Super Class</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Super Class</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Super Class</em>' reference.
	 * @see #setSuperClass(ProcessClass)
	 * @see poosl.PooslPackage#getProcessClass_SuperClass()
	 * @model
	 * @generated
	 */
	ProcessClass getSuperClass();

	/**
	 * Sets the value of the '{@link poosl.ProcessClass#getSuperClass <em>Super Class</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Super Class</em>' reference.
	 * @see #getSuperClass()
	 * @generated
	 */
	void setSuperClass(ProcessClass value);

	/**
	 * Returns the value of the '<em><b>Parameters</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Parameter}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameters</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameters</em>' containment reference list.
	 * @see poosl.PooslPackage#getProcessClass_Parameters()
	 * @model containment="true"
	 * @generated
	 */
	EList<Parameter> getParameters();

	/**
	 * Returns the value of the '<em><b>Ports</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Port}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Ports</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Ports</em>' containment reference list.
	 * @see poosl.PooslPackage#getProcessClass_Ports()
	 * @model containment="true"
	 * @generated
	 */
	EList<Port> getPorts();

	/**
	 * Returns the value of the '<em><b>Initial Method Call</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Initial Method Call</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Initial Method Call</em>' containment reference.
	 * @see #setInitialMethodCall(ProcessMethodCall)
	 * @see poosl.PooslPackage#getProcessClass_InitialMethodCall()
	 * @model containment="true" required="true"
	 * @generated
	 */
	ProcessMethodCall getInitialMethodCall();

	/**
	 * Sets the value of the '{@link poosl.ProcessClass#getInitialMethodCall <em>Initial Method Call</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Initial Method Call</em>' containment reference.
	 * @see #getInitialMethodCall()
	 * @generated
	 */
	void setInitialMethodCall(ProcessMethodCall value);

} // ProcessClass
