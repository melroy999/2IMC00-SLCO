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
 * A representation of the model object '<em><b>Model</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.Model#getName <em>Name</em>}</li>
 *   <li>{@link poosl.Model#getClasses <em>Classes</em>}</li>
 *   <li>{@link poosl.Model#getProcessInstances <em>Process Instances</em>}</li>
 *   <li>{@link poosl.Model#getChannels <em>Channels</em>}</li>
 *   <li>{@link poosl.Model#getPrimitiveTypes <em>Primitive Types</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getModel()
 * @model
 * @generated
 */
public interface Model extends EObject {
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
	 * @see poosl.PooslPackage#getModel_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link poosl.Model#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Classes</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Class}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Classes</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Classes</em>' containment reference list.
	 * @see poosl.PooslPackage#getModel_Classes()
	 * @model containment="true"
	 * @generated
	 */
	EList<poosl.Class> getClasses();

	/**
	 * Returns the value of the '<em><b>Process Instances</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.ProcessInstance}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Process Instances</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Process Instances</em>' containment reference list.
	 * @see poosl.PooslPackage#getModel_ProcessInstances()
	 * @model containment="true"
	 * @generated
	 */
	EList<ProcessInstance> getProcessInstances();

	/**
	 * Returns the value of the '<em><b>Channels</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Channel}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Channels</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Channels</em>' containment reference list.
	 * @see poosl.PooslPackage#getModel_Channels()
	 * @model containment="true"
	 * @generated
	 */
	EList<Channel> getChannels();

	/**
	 * Returns the value of the '<em><b>Primitive Types</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.PrimitiveType}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Primitive Types</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Primitive Types</em>' containment reference list.
	 * @see poosl.PooslPackage#getModel_PrimitiveTypes()
	 * @model containment="true"
	 * @generated
	 */
	EList<PrimitiveType> getPrimitiveTypes();

} // Model
