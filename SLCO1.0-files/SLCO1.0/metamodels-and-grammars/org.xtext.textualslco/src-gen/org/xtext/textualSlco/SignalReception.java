/**
 */
package org.xtext.textualSlco;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Signal Reception</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.SignalReception#getSignalName <em>Signal Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.SignalReception#getArguments <em>Arguments</em>}</li>
 *   <li>{@link org.xtext.textualSlco.SignalReception#getCondition <em>Condition</em>}</li>
 *   <li>{@link org.xtext.textualSlco.SignalReception#getPort <em>Port</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getSignalReception()
 * @model
 * @generated
 */
public interface SignalReception extends Statement
{
  /**
   * Returns the value of the '<em><b>Signal Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Signal Name</em>' attribute isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Signal Name</em>' attribute.
   * @see #setSignalName(String)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getSignalReception_SignalName()
   * @model
   * @generated
   */
  String getSignalName();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.SignalReception#getSignalName <em>Signal Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Signal Name</em>' attribute.
   * @see #getSignalName()
   * @generated
   */
  void setSignalName(String value);

  /**
   * Returns the value of the '<em><b>Arguments</b></em>' containment reference list.
   * The list contents are of type {@link org.xtext.textualSlco.SignalArgument}.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Arguments</em>' containment reference list isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Arguments</em>' containment reference list.
   * @see org.xtext.textualSlco.TextualSlcoPackage#getSignalReception_Arguments()
   * @model containment="true"
   * @generated
   */
  EList<SignalArgument> getArguments();

  /**
   * Returns the value of the '<em><b>Condition</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Condition</em>' containment reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Condition</em>' containment reference.
   * @see #setCondition(Expression)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getSignalReception_Condition()
   * @model containment="true"
   * @generated
   */
  Expression getCondition();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.SignalReception#getCondition <em>Condition</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Condition</em>' containment reference.
   * @see #getCondition()
   * @generated
   */
  void setCondition(Expression value);

  /**
   * Returns the value of the '<em><b>Port</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Port</em>' containment reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Port</em>' containment reference.
   * @see #setPort(PortReference)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getSignalReception_Port()
   * @model containment="true"
   * @generated
   */
  PortReference getPort();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.SignalReception#getPort <em>Port</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Port</em>' containment reference.
   * @see #getPort()
   * @generated
   */
  void setPort(PortReference value);

} // SignalReception
