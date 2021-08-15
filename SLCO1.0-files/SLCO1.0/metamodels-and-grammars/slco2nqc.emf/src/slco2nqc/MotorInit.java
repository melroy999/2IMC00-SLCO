/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc;

import nqc.OutputPortNameEnum;

import org.eclipse.emf.ecore.EObject;

import slco.Port;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Motor Init</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco2nqc.MotorInit#getPort <em>Port</em>}</li>
 *   <li>{@link slco2nqc.MotorInit#getPower <em>Power</em>}</li>
 *   <li>{@link slco2nqc.MotorInit#getOutputName <em>Output Name</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco2nqc.Slco2nqcPackage#getMotorInit()
 * @model
 * @generated
 */
public interface MotorInit extends EObject {
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
	 * @see slco2nqc.Slco2nqcPackage#getMotorInit_Port()
	 * @model required="true"
	 * @generated
	 */
	Port getPort();

	/**
	 * Sets the value of the '{@link slco2nqc.MotorInit#getPort <em>Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Port</em>' reference.
	 * @see #getPort()
	 * @generated
	 */
	void setPort(Port value);

	/**
	 * Returns the value of the '<em><b>Power</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Power</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Power</em>' attribute.
	 * @see #setPower(int)
	 * @see slco2nqc.Slco2nqcPackage#getMotorInit_Power()
	 * @model required="true"
	 * @generated
	 */
	int getPower();

	/**
	 * Sets the value of the '{@link slco2nqc.MotorInit#getPower <em>Power</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Power</em>' attribute.
	 * @see #getPower()
	 * @generated
	 */
	void setPower(int value);

	/**
	 * Returns the value of the '<em><b>Output Name</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.OutputPortNameEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Output Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Output Name</em>' attribute.
	 * @see nqc.OutputPortNameEnum
	 * @see #setOutputName(OutputPortNameEnum)
	 * @see slco2nqc.Slco2nqcPackage#getMotorInit_OutputName()
	 * @model
	 * @generated
	 */
	OutputPortNameEnum getOutputName();

	/**
	 * Sets the value of the '{@link slco2nqc.MotorInit#getOutputName <em>Output Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Output Name</em>' attribute.
	 * @see nqc.OutputPortNameEnum
	 * @see #getOutputName()
	 * @generated
	 */
	void setOutputName(OutputPortNameEnum value);

} // MotorInit
