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
 * A representation of the model object '<em><b>Program</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.Program#getName <em>Name</em>}</li>
 *   <li>{@link nqc.Program#getTasks <em>Tasks</em>}</li>
 *   <li>{@link nqc.Program#getFunctions <em>Functions</em>}</li>
 *   <li>{@link nqc.Program#getSubroutines <em>Subroutines</em>}</li>
 *   <li>{@link nqc.Program#getGlobalVariables <em>Global Variables</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getProgram()
 * @model
 * @generated
 */
public interface Program extends EObject {
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
	 * @see nqc.NqcPackage#getProgram_Name()
	 * @model unique="false" required="true" ordered="false"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link nqc.Program#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Tasks</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.Task}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Tasks</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Tasks</em>' containment reference list.
	 * @see nqc.NqcPackage#getProgram_Tasks()
	 * @model containment="true" ordered="false"
	 * @generated
	 */
	EList<Task> getTasks();

	/**
	 * Returns the value of the '<em><b>Functions</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.Function}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Functions</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Functions</em>' containment reference list.
	 * @see nqc.NqcPackage#getProgram_Functions()
	 * @model containment="true" ordered="false"
	 * @generated
	 */
	EList<Function> getFunctions();

	/**
	 * Returns the value of the '<em><b>Subroutines</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.Subroutine}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Subroutines</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Subroutines</em>' containment reference list.
	 * @see nqc.NqcPackage#getProgram_Subroutines()
	 * @model containment="true" ordered="false"
	 * @generated
	 */
	EList<Subroutine> getSubroutines();

	/**
	 * Returns the value of the '<em><b>Global Variables</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.Variable}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Global Variables</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Global Variables</em>' containment reference list.
	 * @see nqc.NqcPackage#getProgram_GlobalVariables()
	 * @model containment="true" upper="32" ordered="false"
	 * @generated
	 */
	EList<Variable> getGlobalVariables();

} // Program
