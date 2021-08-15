/**
 */
package org.xtext.textualSlco;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>State Machine</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.StateMachine#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.StateMachine#getVariables <em>Variables</em>}</li>
 *   <li>{@link org.xtext.textualSlco.StateMachine#getVertices <em>Vertices</em>}</li>
 *   <li>{@link org.xtext.textualSlco.StateMachine#getTransitions <em>Transitions</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getStateMachine()
 * @model
 * @generated
 */
public interface StateMachine extends EObject
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
   * @see org.xtext.textualSlco.TextualSlcoPackage#getStateMachine_Name()
   * @model
   * @generated
   */
  String getName();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.StateMachine#getName <em>Name</em>}' attribute.
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
   * @see org.xtext.textualSlco.TextualSlcoPackage#getStateMachine_Variables()
   * @model containment="true"
   * @generated
   */
  EList<Variable> getVariables();

  /**
   * Returns the value of the '<em><b>Vertices</b></em>' containment reference list.
   * The list contents are of type {@link org.xtext.textualSlco.Vertex}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Vertices</em>' containment reference list isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Vertices</em>' containment reference list.
   * @see org.xtext.textualSlco.TextualSlcoPackage#getStateMachine_Vertices()
   * @model containment="true"
   * @generated
   */
  EList<Vertex> getVertices();

  /**
   * Returns the value of the '<em><b>Transitions</b></em>' containment reference list.
   * The list contents are of type {@link org.xtext.textualSlco.Transition}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Transitions</em>' containment reference list isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Transitions</em>' containment reference list.
   * @see org.xtext.textualSlco.TextualSlcoPackage#getStateMachine_Transitions()
   * @model containment="true"
   * @generated
   */
  EList<Transition> getTransitions();

} // StateMachine
