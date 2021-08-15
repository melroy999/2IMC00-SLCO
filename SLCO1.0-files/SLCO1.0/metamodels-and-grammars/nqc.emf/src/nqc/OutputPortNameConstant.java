/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Output Port Name Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.OutputPortNameConstant#getOutputPortName <em>Output Port Name</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getOutputPortNameConstant()
 * @model
 * @generated
 */
public interface OutputPortNameConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Output Port Name</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.OutputPortNameEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Output Port Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Output Port Name</em>' attribute.
	 * @see nqc.OutputPortNameEnum
	 * @see #setOutputPortName(OutputPortNameEnum)
	 * @see nqc.NqcPackage#getOutputPortNameConstant_OutputPortName()
	 * @model required="true"
	 * @generated
	 */
	OutputPortNameEnum getOutputPortName();

	/**
	 * Sets the value of the '{@link nqc.OutputPortNameConstant#getOutputPortName <em>Output Port Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Output Port Name</em>' attribute.
	 * @see nqc.OutputPortNameEnum
	 * @see #getOutputPortName()
	 * @generated
	 */
	void setOutputPortName(OutputPortNameEnum value);

} // OutputPortNameConstant
