/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Message Send</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.MessageSend#getMessage <em>Message</em>}</li>
 *   <li>{@link poosl.MessageSend#getPort <em>Port</em>}</li>
 *   <li>{@link poosl.MessageSend#getExpressions <em>Expressions</em>}</li>
 *   <li>{@link poosl.MessageSend#getExpressionLists <em>Expression Lists</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getMessageSend()
 * @model
 * @generated
 */
public interface MessageSend extends Statement {
	/**
	 * Returns the value of the '<em><b>Message</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Message</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Message</em>' containment reference.
	 * @see #setMessage(OutgoingMessage)
	 * @see poosl.PooslPackage#getMessageSend_Message()
	 * @model containment="true" required="true"
	 * @generated
	 */
	OutgoingMessage getMessage();

	/**
	 * Sets the value of the '{@link poosl.MessageSend#getMessage <em>Message</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Message</em>' containment reference.
	 * @see #getMessage()
	 * @generated
	 */
	void setMessage(OutgoingMessage value);

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
	 * @see poosl.PooslPackage#getMessageSend_Port()
	 * @model required="true"
	 * @generated
	 */
	Port getPort();

	/**
	 * Sets the value of the '{@link poosl.MessageSend#getPort <em>Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Port</em>' reference.
	 * @see #getPort()
	 * @generated
	 */
	void setPort(Port value);

	/**
	 * Returns the value of the '<em><b>Expressions</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Expressions</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Expressions</em>' containment reference list.
	 * @see poosl.PooslPackage#getMessageSend_Expressions()
	 * @model containment="true"
	 * @generated
	 */
	EList<Expression> getExpressions();

	/**
	 * Returns the value of the '<em><b>Expression Lists</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.ExpressionList}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Expression Lists</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Expression Lists</em>' containment reference list.
	 * @see poosl.PooslPackage#getMessageSend_ExpressionLists()
	 * @model containment="true"
	 * @generated
	 */
	EList<ExpressionList> getExpressionLists();

} // MessageSend
