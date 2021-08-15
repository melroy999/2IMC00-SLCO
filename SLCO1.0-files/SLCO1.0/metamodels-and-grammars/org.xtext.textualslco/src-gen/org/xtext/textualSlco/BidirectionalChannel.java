/**
 */
package org.xtext.textualSlco;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Bidirectional Channel</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.BidirectionalChannel#getObject1 <em>Object1</em>}</li>
 *   <li>{@link org.xtext.textualSlco.BidirectionalChannel#getPort1 <em>Port1</em>}</li>
 *   <li>{@link org.xtext.textualSlco.BidirectionalChannel#getObject2 <em>Object2</em>}</li>
 *   <li>{@link org.xtext.textualSlco.BidirectionalChannel#getPort2 <em>Port2</em>}</li>
 * </ul>
 * </p>
 *
 * @see org.xtext.textualSlco.TextualSlcoPackage#getBidirectionalChannel()
 * @model
 * @generated
 */
public interface BidirectionalChannel extends Channel
{
  /**
   * Returns the value of the '<em><b>Object1</b></em>' reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Object1</em>' reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Object1</em>' reference.
   * @see #setObject1(org.xtext.textualSlco.Object)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getBidirectionalChannel_Object1()
   * @model
   * @generated
   */
  org.xtext.textualSlco.Object getObject1();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.BidirectionalChannel#getObject1 <em>Object1</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Object1</em>' reference.
   * @see #getObject1()
   * @generated
   */
  void setObject1(org.xtext.textualSlco.Object value);

  /**
   * Returns the value of the '<em><b>Port1</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Port1</em>' containment reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Port1</em>' containment reference.
   * @see #setPort1(PortReference)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getBidirectionalChannel_Port1()
   * @model containment="true"
   * @generated
   */
  PortReference getPort1();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.BidirectionalChannel#getPort1 <em>Port1</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Port1</em>' containment reference.
   * @see #getPort1()
   * @generated
   */
  void setPort1(PortReference value);

  /**
   * Returns the value of the '<em><b>Object2</b></em>' reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Object2</em>' reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Object2</em>' reference.
   * @see #setObject2(org.xtext.textualSlco.Object)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getBidirectionalChannel_Object2()
   * @model
   * @generated
   */
  org.xtext.textualSlco.Object getObject2();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.BidirectionalChannel#getObject2 <em>Object2</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Object2</em>' reference.
   * @see #getObject2()
   * @generated
   */
  void setObject2(org.xtext.textualSlco.Object value);

  /**
   * Returns the value of the '<em><b>Port2</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of the '<em>Port2</em>' containment reference isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @return the value of the '<em>Port2</em>' containment reference.
   * @see #setPort2(PortReference)
   * @see org.xtext.textualSlco.TextualSlcoPackage#getBidirectionalChannel_Port2()
   * @model containment="true"
   * @generated
   */
  PortReference getPort2();

  /**
   * Sets the value of the '{@link org.xtext.textualSlco.BidirectionalChannel#getPort2 <em>Port2</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Port2</em>' containment reference.
   * @see #getPort2()
   * @generated
   */
  void setPort2(PortReference value);

} // BidirectionalChannel
