/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc;

import nqc.OutputModeEnum;

import org.eclipse.emf.ecore.EObject;

import slco.Port;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Signal2 Motor</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.Signal2Motor#getSignalName <em>Signal Name</em>}</li>
 *   <li>{@link slco2nqc.Signal2Motor#getPort <em>Port</em>}</li>
 *   <li>{@link slco2nqc.Signal2Motor#getOutputMode <em>Output Mode</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getSignal2Motor()
 * @model
 * @generated
 */
public interface Signal2Motor extends EObject {
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
	 * @see slco2nqc.Slco2nqcPackage#getSignal2Motor_SignalName()
	 * @model required="true"
	 * @generated
	 */
	String getSignalName();

	/**
	 * Sets the value of the '{@link slco2nqc.Signal2Motor#getSignalName <em>Signal Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Signal Name</em>' attribute.
	 * @see #getSignalName()
	 * @generated
	 */
	void setSignalName(String value);

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
	 * @see slco2nqc.Slco2nqcPackage#getSignal2Motor_Port()
	 * @model required="true"
	 * @generated
	 */
	Port getPort();

	/**
	 * Sets the value of the '{@link slco2nqc.Signal2Motor#getPort <em>Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Port</em>' reference.
	 * @see #getPort()
	 * @generated
	 */
	void setPort(Port value);

	/**
	 * Returns the value of the '<em><b>Output Mode</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.OutputModeEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Output Mode</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Output Mode</em>' attribute.
	 * @see nqc.OutputModeEnum
	 * @see #setOutputMode(OutputModeEnum)
	 * @see slco2nqc.Slco2nqcPackage#getSignal2Motor_OutputMode()
	 * @model required="true"
	 * @generated
	 */
	OutputModeEnum getOutputMode();

	/**
	 * Sets the value of the '{@link slco2nqc.Signal2Motor#getOutputMode <em>Output Mode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Output Mode</em>' attribute.
	 * @see nqc.OutputModeEnum
	 * @see #getOutputMode()
	 * @generated
	 */
	void setOutputMode(OutputModeEnum value);

} // Signal2Motor
