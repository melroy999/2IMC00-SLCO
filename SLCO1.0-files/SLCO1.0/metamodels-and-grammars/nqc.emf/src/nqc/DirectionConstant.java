/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Direction Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.DirectionConstant#getDirection <em>Direction</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getDirectionConstant()
 * @model
 * @generated
 */
public interface DirectionConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Direction</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.DirectionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Direction</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Direction</em>' attribute.
	 * @see nqc.DirectionEnum
	 * @see #setDirection(DirectionEnum)
	 * @see nqc.NqcPackage#getDirectionConstant_Direction()
	 * @model required="true"
	 * @generated
	 */
	DirectionEnum getDirection();

	/**
	 * Sets the value of the '{@link nqc.DirectionConstant#getDirection <em>Direction</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Direction</em>' attribute.
	 * @see nqc.DirectionEnum
	 * @see #getDirection()
	 * @generated
	 */
	void setDirection(DirectionEnum value);

} // DirectionConstant
