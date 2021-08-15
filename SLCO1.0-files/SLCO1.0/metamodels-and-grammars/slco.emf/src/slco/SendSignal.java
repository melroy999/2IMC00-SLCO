/**
 */
package slco;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Send Signal</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.SendSignal#getPort <em>Port</em>}</li>
 *   <li>{@link slco.SendSignal#getSignalName <em>Signal Name</em>}</li>
 *   <li>{@link slco.SendSignal#getArguments <em>Arguments</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getSendSignal()
 * @model
 * @generated
 */
public interface SendSignal extends Statement {
	/**
	 * Returns the value of the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Port</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Port</em>' reference.
	 * @see #setPort(Port)
	 * @see slco.SlcoPackage#getSendSignal_Port()
	 * @model required="true" ordered="false"
	 * @generated
	 */
	Port getPort();

	/**
	 * Sets the value of the '{@link slco.SendSignal#getPort <em>Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Port</em>' reference.
	 * @see #getPort()
	 * @generated
	 */
	void setPort(Port value);

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
	 * @see slco.SlcoPackage#getSendSignal_SignalName()
	 * @model required="true"
	 * @generated
	 */
	String getSignalName();

	/**
	 * Sets the value of the '{@link slco.SendSignal#getSignalName <em>Signal Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Signal Name</em>' attribute.
	 * @see #getSignalName()
	 * @generated
	 */
	void setSignalName(String value);

	/**
	 * Returns the value of the '<em><b>Arguments</b></em>' containment reference list.
	 * The list contents are of type {@link slco.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Arguments</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Arguments</em>' containment reference list.
	 * @see slco.SlcoPackage#getSendSignal_Arguments()
	 * @model containment="true"
	 * @generated
	 */
	EList<Expression> getArguments();

} // SendSignal
