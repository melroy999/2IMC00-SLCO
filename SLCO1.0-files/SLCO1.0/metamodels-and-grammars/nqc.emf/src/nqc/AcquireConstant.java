/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Acquire Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.AcquireConstant#getAcquire <em>Acquire</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getAcquireConstant()
 * @model
 * @generated
 */
public interface AcquireConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Acquire</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.AcquireEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Acquire</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Acquire</em>' attribute.
	 * @see nqc.AcquireEnum
	 * @see #setAcquire(AcquireEnum)
	 * @see nqc.NqcPackage#getAcquireConstant_Acquire()
	 * @model required="true"
	 * @generated
	 */
	AcquireEnum getAcquire();

	/**
	 * Sets the value of the '{@link nqc.AcquireConstant#getAcquire <em>Acquire</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Acquire</em>' attribute.
	 * @see nqc.AcquireEnum
	 * @see #getAcquire()
	 * @generated
	 */
	void setAcquire(AcquireEnum value);

} // AcquireConstant
