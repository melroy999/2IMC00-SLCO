/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Serial Channel Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SerialChannelConstant#getSerialChannel <em>Serial Channel</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSerialChannelConstant()
 * @model
 * @generated
 */
public interface SerialChannelConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Serial Channel</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SerialChannelEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Serial Channel</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Serial Channel</em>' attribute.
	 * @see nqc.SerialChannelEnum
	 * @see #setSerialChannel(SerialChannelEnum)
	 * @see nqc.NqcPackage#getSerialChannelConstant_SerialChannel()
	 * @model required="true"
	 * @generated
	 */
	SerialChannelEnum getSerialChannel();

	/**
	 * Sets the value of the '{@link nqc.SerialChannelConstant#getSerialChannel <em>Serial Channel</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Serial Channel</em>' attribute.
	 * @see nqc.SerialChannelEnum
	 * @see #getSerialChannel()
	 * @generated
	 */
	void setSerialChannel(SerialChannelEnum value);

} // SerialChannelConstant
