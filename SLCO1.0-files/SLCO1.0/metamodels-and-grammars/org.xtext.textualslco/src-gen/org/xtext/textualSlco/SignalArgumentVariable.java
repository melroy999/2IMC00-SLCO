/**
 */
package org.xtext.textualSlco;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Signal Argument Variable</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.SignalArgumentVariable#getVariable <em>Variable</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getSignalArgumentVariable()
 * @model
 * @generated
 */
public interface SignalArgumentVariable extends SignalArgument
{
  /**
   * Returns the value of the '<em><b>Variable</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Variable</em>' containment reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Variable</em>' containment reference.
   * @see #setVariable(VariableReference)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getSignalArgumentVariable_Variable()
   * @model containment="true"
   * @generated
   */
  VariableReference getVariable();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.SignalArgumentVariable#getVariable <em>Variable</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Variable</em>' containment reference.
   * @see #getVariable()
   * @generated
   */
  void setVariable(VariableReference value);

} // SignalArgumentVariable
