/**
 */
package org.xtext.textualSlco;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Variable</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.Variable#getType <em>Type</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Variable#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Variable#getIntialValue <em>Intial Value</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getVariable()
 * @model
 * @generated
 */
public interface Variable extends EObject
{
  /**
   * Returns the value of the '<em><b>Type</b></em>' attribute.
   * The literals are from the enumeration {@link org.xtext.textualSlco.PrimitiveType}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Type</em>' attribute isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Type</em>' attribute.
   * @see org.xtext.textualSlco.PrimitiveType
   * @see #setType(PrimitiveType)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getVariable_Type()
   * @model
   * @generated
   */
  PrimitiveType getType();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.Variable#getType <em>Type</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Type</em>' attribute.
   * @see org.xtext.textualSlco.PrimitiveType
   * @see #getType()
   * @generated
   */
  void setType(PrimitiveType value);

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
   * @see org.xtext.textualSlco.TextualSlcoPackage#getVariable_Name()
   * @model
   * @generated
   */
  String getName();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.Variable#getName <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Name</em>' attribute.
   * @see #getName()
   * @generated
   */
  void setName(String value);

  /**
   * Returns the value of the '<em><b>Intial Value</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Intial Value</em>' containment reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Intial Value</em>' containment reference.
   * @see #setIntialValue(Expression)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getVariable_IntialValue()
   * @model containment="true"
   * @generated
   */
  Expression getIntialValue();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.Variable#getIntialValue <em>Intial Value</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Intial Value</em>' containment reference.
   * @see #getIntialValue()
   * @generated
   */
  void setIntialValue(Expression value);

} // Variable
