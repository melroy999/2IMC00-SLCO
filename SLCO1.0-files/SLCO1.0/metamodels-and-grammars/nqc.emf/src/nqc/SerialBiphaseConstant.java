/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Serial Biphase Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SerialBiphaseConstant#getSerialBiphase <em>Serial Biphase</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSerialBiphaseConstant()
 * @model
 * @generated
 */
public interface SerialBiphaseConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Serial Biphase</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SerialBiphaseEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Serial Biphase</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Serial Biphase</em>' attribute.
	 * @see nqc.SerialBiphaseEnum
	 * @see #setSerialBiphase(SerialBiphaseEnum)
	 * @see nqc.NqcPackage#getSerialBiphaseConstant_SerialBiphase()
	 * @model required="true"
	 * @generated
	 */
	SerialBiphaseEnum getSerialBiphase();

	/**
	 * Sets the value of the '{@link nqc.SerialBiphaseConstant#getSerialBiphase <em>Serial Biphase</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Serial Biphase</em>' attribute.
	 * @see nqc.SerialBiphaseEnum
	 * @see #getSerialBiphase()
	 * @generated
	 */
	void setSerialBiphase(SerialBiphaseEnum value);

} // SerialBiphaseConstant
