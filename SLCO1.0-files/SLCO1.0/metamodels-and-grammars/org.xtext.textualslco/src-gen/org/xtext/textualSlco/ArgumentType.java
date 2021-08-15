/**
 */
package org.xtext.textualSlco;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Argument Type</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.ArgumentType#getType <em>Type</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getArgumentType()
 * @model
 * @generated
 */
public interface ArgumentType extends EObject
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
   * @see org.xtext.textualSlco.TextualSlcoPackage#getArgumentType_Type()
   * @model
   * @generated
   */
  PrimitiveType getType();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.ArgumentType#getType <em>Type</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Type</em>' attribute.
   * @see org.xtext.textualSlco.PrimitiveType
   * @see #getType()
   * @generated
   */
  void setType(PrimitiveType value);

} // ArgumentType
