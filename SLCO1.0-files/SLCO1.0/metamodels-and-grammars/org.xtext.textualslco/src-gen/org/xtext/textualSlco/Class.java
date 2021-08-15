/**
 */
package org.xtext.textualSlco;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Class</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.Class#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Class#getVariables <em>Variables</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Class#getPorts <em>Ports</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Class#getStateMachines <em>State Machines</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getClass_()
 * @model
 * @generated
 */
public interface Class extends EObject
{
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
   * @see org.xtext.textualSlco.TextualSlcoPackage#getClass_Name()
   * @model
   * @generated
   */
  String getName();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.Class#getName <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Name</em>' attribute.
   * @see #getName()
   * @generated
   */
  void setName(String value);

  /**
   * Returns the value of the '<em><b>Variables</b></em>' containment reference list.
   * The list contents are of type {@link org.xtext.textualSlco.Variable}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Variables</em>' containment reference list isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Variables</em>' containment reference list.
   * @see org.xtext.textualSlco.TextualSlcoPackage#getClass_Variables()
   * @model containment="true"
   * @generated
   */
  EList<Variable> getVariables();

  /**
   * Returns the value of the '<em><b>Ports</b></em>' containment reference list.
   * The list contents are of type {@link org.xtext.textualSlco.Port}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Ports</em>' containment reference list isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Ports</em>' containment reference list.
   * @see org.xtext.textualSlco.TextualSlcoPackage#getClass_Ports()
   * @model containment="true"
   * @generated
   */
  EList<Port> getPorts();

  /**
   * Returns the value of the '<em><b>State Machines</b></em>' containment reference list.
   * The list contents are of type {@link org.xtext.textualSlco.StateMachine}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>State Machines</em>' containment reference list isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>State Machines</em>' containment reference list.
   * @see org.xtext.textualSlco.TextualSlcoPackage#getClass_StateMachines()
   * @model containment="true"
   * @generated
   */
  EList<StateMachine> getStateMachines();

} // Class
