/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Serial Checksum Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SerialChecksumConstant#getSerialChecksum <em>Serial Checksum</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSerialChecksumConstant()
 * @model
 * @generated
 */
public interface SerialChecksumConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Serial Checksum</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SerialChecksumEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Serial Checksum</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Serial Checksum</em>' attribute.
	 * @see nqc.SerialChecksumEnum
	 * @see #setSerialChecksum(SerialChecksumEnum)
	 * @see nqc.NqcPackage#getSerialChecksumConstant_SerialChecksum()
	 * @model required="true"
	 * @generated
	 */
	SerialChecksumEnum getSerialChecksum();

	/**
	 * Sets the value of the '{@link nqc.SerialChecksumConstant#getSerialChecksum <em>Serial Checksum</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Serial Checksum</em>' attribute.
	 * @see nqc.SerialChecksumEnum
	 * @see #getSerialChecksum()
	 * @generated
	 */
	void setSerialChecksum(SerialChecksumEnum value);

} // SerialChecksumConstant
