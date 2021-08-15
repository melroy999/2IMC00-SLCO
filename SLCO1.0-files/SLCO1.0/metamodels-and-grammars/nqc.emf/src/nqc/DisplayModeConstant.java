/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Display Mode Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.DisplayModeConstant#getDisplayMode <em>Display Mode</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getDisplayModeConstant()
 * @model
 * @generated
 */
public interface DisplayModeConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Display Mode</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.DisplayModeEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Display Mode</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Display Mode</em>' attribute.
	 * @see nqc.DisplayModeEnum
	 * @see #setDisplayMode(DisplayModeEnum)
	 * @see nqc.NqcPackage#getDisplayModeConstant_DisplayMode()
	 * @model required="true"
	 * @generated
	 */
	DisplayModeEnum getDisplayMode();

	/**
	 * Sets the value of the '{@link nqc.DisplayModeConstant#getDisplayMode <em>Display Mode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Display Mode</em>' attribute.
	 * @see nqc.DisplayModeEnum
	 * @see #getDisplayMode()
	 * @generated
	 */
	void setDisplayMode(DisplayModeEnum value);

} // DisplayModeConstant
