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
 * A representation of the model object '<em><b>Channel</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.Channel#getName <em>Name</em>}</li>
 *   <li>{@link poosl.Channel#getPort1 <em>Port1</em>}</li>
 *   <li>{@link poosl.Channel#getPort2 <em>Port2</em>}</li>
 *   <li>{@link poosl.Channel#getProcessInstance1 <em>Process Instance1</em>}</li>
 *   <li>{@link poosl.Channel#getProcessInstance2 <em>Process Instance2</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getChannel()
 * @model
 * @generated
 */
public interface Channel extends EObject {
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
	 * @see poosl.PooslPackage#getChannel_Name()
	 * @model required="true"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link poosl.Channel#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Port1</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Port1</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Port1</em>' reference.
	 * @see #setPort1(Port)
	 * @see poosl.PooslPackage#getChannel_Port1()
	 * @model required="true"
	 * @generated
	 */
	Port getPort1();

	/**
	 * Sets the value of the '{@link poosl.Channel#getPort1 <em>Port1</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Port1</em>' reference.
	 * @see #getPort1()
	 * @generated
	 */
	void setPort1(Port value);

	/**
	 * Returns the value of the '<em><b>Port2</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Port2</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Port2</em>' reference.
	 * @see #setPort2(Port)
	 * @see poosl.PooslPackage#getChannel_Port2()
	 * @model required="true"
	 * @generated
	 */
	Port getPort2();

	/**
	 * Sets the value of the '{@link poosl.Channel#getPort2 <em>Port2</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Port2</em>' reference.
	 * @see #getPort2()
	 * @generated
	 */
	void setPort2(Port value);

	/**
	 * Returns the value of the '<em><b>Process Instance1</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Process Instance1</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Process Instance1</em>' reference.
	 * @see #setProcessInstance1(ProcessInstance)
	 * @see poosl.PooslPackage#getChannel_ProcessInstance1()
	 * @model required="true"
	 * @generated
	 */
	ProcessInstance getProcessInstance1();

	/**
	 * Sets the value of the '{@link poosl.Channel#getProcessInstance1 <em>Process Instance1</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Process Instance1</em>' reference.
	 * @see #getProcessInstance1()
	 * @generated
	 */
	void setProcessInstance1(ProcessInstance value);

	/**
	 * Returns the value of the '<em><b>Process Instance2</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Process Instance2</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Process Instance2</em>' reference.
	 * @see #setProcessInstance2(ProcessInstance)
	 * @see poosl.PooslPackage#getChannel_ProcessInstance2()
	 * @model required="true"
	 * @generated
	 */
	ProcessInstance getProcessInstance2();

	/**
	 * Sets the value of the '{@link poosl.Channel#getProcessInstance2 <em>Process Instance2</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Process Instance2</em>' reference.
	 * @see #getProcessInstance2()
	 * @generated
	 */
	void setProcessInstance2(ProcessInstance value);

} // Channel
