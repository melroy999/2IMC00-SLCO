/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Nullary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.NullaryBuiltInFunctionCall#getNullaryBuiltInFunction <em>Nullary Built In Function</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getNullaryBuiltInFunctionCall()
 * @model
 * @generated
 */
public interface NullaryBuiltInFunctionCall extends BuiltInFunctionCall {
	/**
	 * Returns the value of the '<em><b>Nullary Built In Function</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.BuiltInNullaryFunctionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Nullary Built In Function</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Nullary Built In Function</em>' attribute.
	 * @see nqc.BuiltInNullaryFunctionEnum
	 * @see #setNullaryBuiltInFunction(BuiltInNullaryFunctionEnum)
	 * @see nqc.NqcPackage#getNullaryBuiltInFunctionCall_NullaryBuiltInFunction()
	 * @model required="true"
	 * @generated
	 */
	BuiltInNullaryFunctionEnum getNullaryBuiltInFunction();

	/**
	 * Sets the value of the '{@link nqc.NullaryBuiltInFunctionCall#getNullaryBuiltInFunction <em>Nullary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Nullary Built In Function</em>' attribute.
	 * @see nqc.BuiltInNullaryFunctionEnum
	 * @see #getNullaryBuiltInFunction()
	 * @generated
	 */
	void setNullaryBuiltInFunction(BuiltInNullaryFunctionEnum value);

} // NullaryBuiltInFunctionCall
