/**
 */
package slco;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Bidirectional Channel</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.BidirectionalChannel#getPort1 <em>Port1</em>}</li>
 *   <li>{@link slco.BidirectionalChannel#getPort2 <em>Port2</em>}</li>
 *   <li>{@link slco.BidirectionalChannel#getObject1 <em>Object1</em>}</li>
 *   <li>{@link slco.BidirectionalChannel#getObject2 <em>Object2</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getBidirectionalChannel()
 * @model
 * @generated
 */
public interface BidirectionalChannel extends Channel {
	/**
	 * Returns the value of the '<em><b>Port1</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Port1</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Port1</em>' reference.
	 * @see #setPort1(Port)
	 * @see slco.SlcoPackage#getBidirectionalChannel_Port1()
	 * @model required="true"
	 * @generated
	 */
	Port getPort1();

	/**
	 * Sets the value of the '{@link slco.BidirectionalChannel#getPort1 <em>Port1</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Port1</em>' reference.
	 * @see #getPort1()
	 * @generated
	 */
	void setPort1(Port value);

	/**
	 * Returns the value of the '<em><b>Port2</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Port2</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Port2</em>' reference.
	 * @see #setPort2(Port)
	 * @see slco.SlcoPackage#getBidirectionalChannel_Port2()
	 * @model required="true"
	 * @generated
	 */
	Port getPort2();

	/**
	 * Sets the value of the '{@link slco.BidirectionalChannel#getPort2 <em>Port2</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Port2</em>' reference.
	 * @see #getPort2()
	 * @generated
	 */
	void setPort2(Port value);

	/**
	 * Returns the value of the '<em><b>Object1</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Object1</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Object1</em>' reference.
	 * @see #setObject1(slco.Object)
	 * @see slco.SlcoPackage#getBidirectionalChannel_Object1()
	 * @model required="true"
	 * @generated
	 */
	slco.Object getObject1();

	/**
	 * Sets the value of the '{@link slco.BidirectionalChannel#getObject1 <em>Object1</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Object1</em>' reference.
	 * @see #getObject1()
	 * @generated
	 */
	void setObject1(slco.Object value);

	/**
	 * Returns the value of the '<em><b>Object2</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Object2</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Object2</em>' reference.
	 * @see #setObject2(slco.Object)
	 * @see slco.SlcoPackage#getBidirectionalChannel_Object2()
	 * @model required="true"
	 * @generated
	 */
	slco.Object getObject2();

	/**
	 * Sets the value of the '{@link slco.BidirectionalChannel#getObject2 <em>Object2</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Object2</em>' reference.
	 * @see #getObject2()
	 * @generated
	 */
	void setObject2(slco.Object value);

} // BidirectionalChannel
