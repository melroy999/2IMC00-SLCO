/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Serial Comm Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SerialCommConstant#getSerialComm <em>Serial Comm</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSerialCommConstant()
 * @model
 * @generated
 */
public interface SerialCommConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Serial Comm</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SerialCommEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Serial Comm</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Serial Comm</em>' attribute.
	 * @see nqc.SerialCommEnum
	 * @see #setSerialComm(SerialCommEnum)
	 * @see nqc.NqcPackage#getSerialCommConstant_SerialComm()
	 * @model required="true"
	 * @generated
	 */
	SerialCommEnum getSerialComm();

	/**
	 * Sets the value of the '{@link nqc.SerialCommConstant#getSerialComm <em>Serial Comm</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Serial Comm</em>' attribute.
	 * @see nqc.SerialCommEnum
	 * @see #getSerialComm()
	 * @generated
	 */
	void setSerialComm(SerialCommEnum value);

} // SerialCommConstant
