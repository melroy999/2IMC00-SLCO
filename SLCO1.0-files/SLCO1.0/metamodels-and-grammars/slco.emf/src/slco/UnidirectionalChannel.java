/**
 */
package slco;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Unidirectional Channel</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.UnidirectionalChannel#getSourcePort <em>Source Port</em>}</li>
 *   <li>{@link slco.UnidirectionalChannel#getTargetPort <em>Target Port</em>}</li>
 *   <li>{@link slco.UnidirectionalChannel#getSourceObject <em>Source Object</em>}</li>
 *   <li>{@link slco.UnidirectionalChannel#getTargetObject <em>Target Object</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getUnidirectionalChannel()
 * @model
 * @generated
 */
public interface UnidirectionalChannel extends Channel {
	/**
	 * Returns the value of the '<em><b>Source Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Source Port</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Source Port</em>' reference.
	 * @see #setSourcePort(Port)
	 * @see slco.SlcoPackage#getUnidirectionalChannel_SourcePort()
	 * @model required="true"
	 * @generated
	 */
	Port getSourcePort();

	/**
	 * Sets the value of the '{@link slco.UnidirectionalChannel#getSourcePort <em>Source Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Source Port</em>' reference.
	 * @see #getSourcePort()
	 * @generated
	 */
	void setSourcePort(Port value);

	/**
	 * Returns the value of the '<em><b>Target Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Target Port</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Target Port</em>' reference.
	 * @see #setTargetPort(Port)
	 * @see slco.SlcoPackage#getUnidirectionalChannel_TargetPort()
	 * @model required="true"
	 * @generated
	 */
	Port getTargetPort();

	/**
	 * Sets the value of the '{@link slco.UnidirectionalChannel#getTargetPort <em>Target Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Target Port</em>' reference.
	 * @see #getTargetPort()
	 * @generated
	 */
	void setTargetPort(Port value);

	/**
	 * Returns the value of the '<em><b>Source Object</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Source Object</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Source Object</em>' reference.
	 * @see #setSourceObject(slco.Object)
	 * @see slco.SlcoPackage#getUnidirectionalChannel_SourceObject()
	 * @model required="true"
	 * @generated
	 */
	slco.Object getSourceObject();

	/**
	 * Sets the value of the '{@link slco.UnidirectionalChannel#getSourceObject <em>Source Object</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Source Object</em>' reference.
	 * @see #getSourceObject()
	 * @generated
	 */
	void setSourceObject(slco.Object value);

	/**
	 * Returns the value of the '<em><b>Target Object</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Target Object</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Target Object</em>' reference.
	 * @see #setTargetObject(slco.Object)
	 * @see slco.SlcoPackage#getUnidirectionalChannel_TargetObject()
	 * @model required="true"
	 * @generated
	 */
	slco.Object getTargetObject();

	/**
	 * Sets the value of the '{@link slco.UnidirectionalChannel#getTargetObject <em>Target Object</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Target Object</em>' reference.
	 * @see #getTargetObject()
	 * @generated
	 */
	void setTargetObject(slco.Object value);

} // UnidirectionalChannel
