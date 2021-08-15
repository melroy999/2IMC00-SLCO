/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Sound Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SoundConstant#getSound <em>Sound</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSoundConstant()
 * @model
 * @generated
 */
public interface SoundConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Sound</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.SoundEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sound</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sound</em>' attribute.
	 * @see nqc.SoundEnum
	 * @see #setSound(SoundEnum)
	 * @see nqc.NqcPackage#getSoundConstant_Sound()
	 * @model required="true"
	 * @generated
	 */
	SoundEnum getSound();

	/**
	 * Sets the value of the '{@link nqc.SoundConstant#getSound <em>Sound</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sound</em>' attribute.
	 * @see nqc.SoundEnum
	 * @see #getSound()
	 * @generated
	 */
	void setSound(SoundEnum value);

} // SoundConstant
