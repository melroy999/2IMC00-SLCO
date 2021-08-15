/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Monitor Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.MonitorStatement#getEvents <em>Events</em>}</li>
 *   <li>{@link nqc.MonitorStatement#getBody <em>Body</em>}</li>
 *   <li>{@link nqc.MonitorStatement#getHandlers <em>Handlers</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getMonitorStatement()
 * @model
 * @generated
 */
public interface MonitorStatement extends Statement {
	/**
	 * Returns the value of the '<em><b>Events</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.EventTypeConstant}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Events</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Events</em>' containment reference list.
	 * @see nqc.NqcPackage#getMonitorStatement_Events()
	 * @model containment="true"
	 * @generated
	 */
	EList<EventTypeConstant> getEvents();

	/**
	 * Returns the value of the '<em><b>Body</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Body</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Body</em>' containment reference.
	 * @see #setBody(Statement)
	 * @see nqc.NqcPackage#getMonitorStatement_Body()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Statement getBody();

	/**
	 * Sets the value of the '{@link nqc.MonitorStatement#getBody <em>Body</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Body</em>' containment reference.
	 * @see #getBody()
	 * @generated
	 */
	void setBody(Statement value);

	/**
	 * Returns the value of the '<em><b>Handlers</b></em>' containment reference list.
	 * The list contents are of type {@link nqc.MonitorHandler}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Handlers</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Handlers</em>' containment reference list.
	 * @see nqc.NqcPackage#getMonitorStatement_Handlers()
	 * @model containment="true"
	 * @generated
	 */
	EList<MonitorHandler> getHandlers();

} // MonitorStatement
