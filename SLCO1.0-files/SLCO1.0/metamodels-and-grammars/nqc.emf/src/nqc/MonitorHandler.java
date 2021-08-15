/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Monitor Handler</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.MonitorHandler#getCatchEvent <em>Catch Event</em>}</li>
 *   <li>{@link nqc.MonitorHandler#getHandler <em>Handler</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getMonitorHandler()
 * @model
 * @generated
 */
public interface MonitorHandler extends EObject {
	/**
	 * Returns the value of the '<em><b>Catch Event</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Catch Event</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Catch Event</em>' containment reference.
	 * @see #setCatchEvent(EventTypeConstant)
	 * @see nqc.NqcPackage#getMonitorHandler_CatchEvent()
	 * @model containment="true"
	 * @generated
	 */
	EventTypeConstant getCatchEvent();

	/**
	 * Sets the value of the '{@link nqc.MonitorHandler#getCatchEvent <em>Catch Event</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Catch Event</em>' containment reference.
	 * @see #getCatchEvent()
	 * @generated
	 */
	void setCatchEvent(EventTypeConstant value);

	/**
	 * Returns the value of the '<em><b>Handler</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Handler</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Handler</em>' containment reference.
	 * @see #setHandler(Statement)
	 * @see nqc.NqcPackage#getMonitorHandler_Handler()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Statement getHandler();

	/**
	 * Sets the value of the '{@link nqc.MonitorHandler#getHandler <em>Handler</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Handler</em>' containment reference.
	 * @see #getHandler()
	 * @generated
	 */
	void setHandler(Statement value);

} // MonitorHandler
