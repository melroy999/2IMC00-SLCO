/**
 */
package org.xtext.textualSlco;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Unidirectional Channel</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.UnidirectionalChannel#getSourceObject <em>Source Object</em>}</li>
 *   <li>{@link org.xtext.textualSlco.UnidirectionalChannel#getSourcePort <em>Source Port</em>}</li>
 *   <li>{@link org.xtext.textualSlco.UnidirectionalChannel#getTargetObject <em>Target Object</em>}</li>
 *   <li>{@link org.xtext.textualSlco.UnidirectionalChannel#getTargetPort <em>Target Port</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getUnidirectionalChannel()
 * @model
 * @generated
 */
public interface UnidirectionalChannel extends Channel
{
  /**
   * Returns the value of the '<em><b>Source Object</b></em>' reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Source Object</em>' reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Source Object</em>' reference.
   * @see #setSourceObject(org.xtext.textualSlco.Object)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getUnidirectionalChannel_SourceObject()
   * @model
   * @generated
   */
  org.xtext.textualSlco.Object getSourceObject();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.UnidirectionalChannel#getSourceObject <em>Source Object</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Source Object</em>' reference.
   * @see #getSourceObject()
   * @generated
   */
  void setSourceObject(org.xtext.textualSlco.Object value);

  /**
   * Returns the value of the '<em><b>Source Port</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Source Port</em>' containment reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Source Port</em>' containment reference.
   * @see #setSourcePort(PortReference)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getUnidirectionalChannel_SourcePort()
   * @model containment="true"
   * @generated
   */
  PortReference getSourcePort();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.UnidirectionalChannel#getSourcePort <em>Source Port</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Source Port</em>' containment reference.
   * @see #getSourcePort()
   * @generated
   */
  void setSourcePort(PortReference value);

  /**
   * Returns the value of the '<em><b>Target Object</b></em>' reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Target Object</em>' reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Target Object</em>' reference.
   * @see #setTargetObject(org.xtext.textualSlco.Object)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getUnidirectionalChannel_TargetObject()
   * @model
   * @generated
   */
  org.xtext.textualSlco.Object getTargetObject();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.UnidirectionalChannel#getTargetObject <em>Target Object</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Target Object</em>' reference.
   * @see #getTargetObject()
   * @generated
   */
  void setTargetObject(org.xtext.textualSlco.Object value);

  /**
   * Returns the value of the '<em><b>Target Port</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Target Port</em>' containment reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Target Port</em>' containment reference.
   * @see #setTargetPort(PortReference)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getUnidirectionalChannel_TargetPort()
   * @model containment="true"
   * @generated
   */
  PortReference getTargetPort();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.UnidirectionalChannel#getTargetPort <em>Target Port</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Target Port</em>' containment reference.
   * @see #getTargetPort()
   * @generated
   */
  void setTargetPort(PortReference value);

} // UnidirectionalChannel
