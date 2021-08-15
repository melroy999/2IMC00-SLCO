/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Output Mode Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.OutputModeConstant#getOutputMode <em>Output Mode</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getOutputModeConstant()
 * @model
 * @generated
 */
public interface OutputModeConstant extends ConstantExpression {
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
	 * @see nqc.NqcPackage#getOutputModeConstant_OutputMode()
	 * @model required="true"
	 * @generated
	 */
	OutputModeEnum getOutputMode();

	/**
	 * Sets the value of the '{@link nqc.OutputModeConstant#getOutputMode <em>Output Mode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Output Mode</em>' attribute.
	 * @see nqc.OutputModeEnum
	 * @see #getOutputMode()
	 * @generated
	 */
	void setOutputMode(OutputModeEnum value);

} // OutputModeConstant
