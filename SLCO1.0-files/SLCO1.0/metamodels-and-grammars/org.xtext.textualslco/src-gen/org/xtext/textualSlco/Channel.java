/**
 */
package org.xtext.textualSlco;

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
 *   <li>{@link org.xtext.textualSlco.Channel#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Channel#getArgumentTypes <em>Argument Types</em>}</li>
 *   <li>{@link org.xtext.textualSlco.Channel#getChannelType <em>Channel Type</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getChannel()
 * @model
 * @generated
 */
public interface Channel extends EObject
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
   * @see org.xtext.textualSlco.TextualSlcoPackage#getChannel_Name()
   * @model
   * @generated
   */
  String getName();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.Channel#getName <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Name</em>' attribute.
   * @see #getName()
   * @generated
   */
  void setName(String value);

  /**
   * Returns the value of the '<em><b>Argument Types</b></em>' containment reference list.
   * The list contents are of type {@link org.xtext.textualSlco.ArgumentType}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Argument Types</em>' containment reference list isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Argument Types</em>' containment reference list.
   * @see org.xtext.textualSlco.TextualSlcoPackage#getChannel_ArgumentTypes()
   * @model containment="true"
   * @generated
   */
  EList<ArgumentType> getArgumentTypes();

  /**
   * Returns the value of the '<em><b>Channel Type</b></em>' attribute.
   * The literals are from the enumeration {@link org.xtext.textualSlco.ChannelType}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Channel Type</em>' attribute isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Channel Type</em>' attribute.
   * @see org.xtext.textualSlco.ChannelType
   * @see #setChannelType(ChannelType)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getChannel_ChannelType()
   * @model
   * @generated
   */
  ChannelType getChannelType();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.Channel#getChannelType <em>Channel Type</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Channel Type</em>' attribute.
   * @see org.xtext.textualSlco.ChannelType
   * @see #getChannelType()
   * @generated
   */
  void setChannelType(ChannelType value);

} // Channel
