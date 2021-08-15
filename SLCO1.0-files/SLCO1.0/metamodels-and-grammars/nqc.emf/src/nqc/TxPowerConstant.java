/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Tx Power Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.TxPowerConstant#getTxPower <em>Tx Power</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getTxPowerConstant()
 * @model
 * @generated
 */
public interface TxPowerConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Tx Power</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.TxPowerEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Tx Power</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Tx Power</em>' attribute.
	 * @see nqc.TxPowerEnum
	 * @see #setTxPower(TxPowerEnum)
	 * @see nqc.NqcPackage#getTxPowerConstant_TxPower()
	 * @model required="true"
	 * @generated
	 */
	TxPowerEnum getTxPower();

	/**
	 * Sets the value of the '{@link nqc.TxPowerConstant#getTxPower <em>Tx Power</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Tx Power</em>' attribute.
	 * @see nqc.TxPowerEnum
	 * @see #getTxPower()
	 * @generated
	 */
	void setTxPower(TxPowerEnum value);

} // TxPowerConstant
