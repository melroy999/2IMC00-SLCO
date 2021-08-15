/**
 */
package slco;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Signal Reception</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.SignalReception#getPort <em>Port</em>}</li>
 *   <li>{@link slco.SignalReception#getSignalName <em>Signal Name</em>}</li>
 *   <li>{@link slco.SignalReception#getArguments <em>Arguments</em>}</li>
 *   <li>{@link slco.SignalReception#getCondition <em>Condition</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getSignalReception()
 * @model
 * @generated
 */
public interface SignalReception extends Statement {
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
	 * @see slco.SlcoPackage#getSignalReception_Port()
	 * @model required="true"
	 * @generated
	 */
	Port getPort();

	/**
	 * Sets the value of the '{@link slco.SignalReception#getPort <em>Port</em>}' reference.
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
	 * @see slco.SlcoPackage#getSignalReception_SignalName()
	 * @model required="true"
	 * @generated
	 */
	String getSignalName();

	/**
	 * Sets the value of the '{@link slco.SignalReception#getSignalName <em>Signal Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Signal Name</em>' attribute.
	 * @see #getSignalName()
	 * @generated
	 */
	void setSignalName(String value);

	/**
	 * Returns the value of the '<em><b>Arguments</b></em>' containment reference list.
	 * The list contents are of type {@link slco.SignalArgument}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Arguments</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Arguments</em>' containment reference list.
	 * @see slco.SlcoPackage#getSignalReception_Arguments()
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
	 * @see slco.SlcoPackage#getSignalReception_Condition()
	 * @model containment="true"
	 * @generated
	 */
	Expression getCondition();

	/**
	 * Sets the value of the '{@link slco.SignalReception#getCondition <em>Condition</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Condition</em>' containment reference.
	 * @see #getCondition()
	 * @generated
	 */
	void setCondition(Expression value);

} // SignalReception
