/**
 */
package org.xtext.textualSlco;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Object</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.Object#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Object#getClass_ <em>Class</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getObject()
 * @model
 * @generated
 */
public interface Object extends EObject
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
   * @see org.xtext.textualSlco.TextualSlcoPackage#getObject_Name()
   * @model
   * @generated
   */
  String getName();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.Object#getName <em>Name</em>}' attribute.
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
   * @see #setClass(org.xtext.textualSlco.Class)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getObject_Class()
   * @model
   * @generated
   */
  org.xtext.textualSlco.Class getClass_();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.Object#getClass_ <em>Class</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Class</em>' reference.
   * @see #getClass_()
   * @generated
   */
  void setClass(org.xtext.textualSlco.Class value);

} // Object
