/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Process Instance</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ProcessInstance#getName <em>Name</em>}</li>
 *   <li>{@link poosl.ProcessInstance#getClass_ <em>Class</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getProcessInstance()
 * @model
 * @generated
 */
public interface ProcessInstance extends EObject {
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
	 * @see poosl.PooslPackage#getProcessInstance_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link poosl.ProcessInstance#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Class</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Class</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Class</em>' reference.
	 * @see #setClass(ProcessClass)
	 * @see poosl.PooslPackage#getProcessInstance_Class()
	 * @model required="true"
	 * @generated
	 */
	ProcessClass getClass_();

	/**
	 * Sets the value of the '{@link poosl.ProcessInstance#getClass_ <em>Class</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Class</em>' reference.
	 * @see #getClass_()
	 * @generated
	 */
	void setClass(ProcessClass value);

} // ProcessInstance
