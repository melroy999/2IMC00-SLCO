/**
 */
package org.xtext.textualSlco;

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
 *   <li>{@link org.xtext.textualSlco.Model#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Model#getClasses <em>Classes</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Model#getObjects <em>Objects</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Model#getChannels <em>Channels</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getModel()
 * @model
 * @generated
 */
public interface Model extends EObject
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
   * @see org.xtext.textualSlco.TextualSlcoPackage#getModel_Name()
   * @model
   * @generated
   */
  String getName();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.Model#getName <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Name</em>' attribute.
   * @see #getName()
   * @generated
   */
  void setName(String value);

  /**
   * Returns the value of the '<em><b>Classes</b></em>' containment reference list.
   * The list contents are of type {@link org.xtext.textualSlco.Class}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Classes</em>' containment reference list isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Classes</em>' containment reference list.
   * @see org.xtext.textualSlco.TextualSlcoPackage#getModel_Classes()
   * @model containment="true"
   * @generated
   */
  EList<org.xtext.textualSlco.Class> getClasses();

  /**
   * Returns the value of the '<em><b>Objects</b></em>' containment reference list.
   * The list contents are of type {@link org.xtext.textualSlco.Object}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Objects</em>' containment reference list isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Objects</em>' containment reference list.
   * @see org.xtext.textualSlco.TextualSlcoPackage#getModel_Objects()
   * @model containment="true"
   * @generated
   */
  EList<org.xtext.textualSlco.Object> getObjects();

  /**
   * Returns the value of the '<em><b>Channels</b></em>' containment reference list.
   * The list contents are of type {@link org.xtext.textualSlco.Channel}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Channels</em>' containment reference list isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Channels</em>' containment reference list.
   * @see org.xtext.textualSlco.TextualSlcoPackage#getModel_Channels()
   * @model containment="true"
   * @generated
   */
  EList<Channel> getChannels();

} // Model
