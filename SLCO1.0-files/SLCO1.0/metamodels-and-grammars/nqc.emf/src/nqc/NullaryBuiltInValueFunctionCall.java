/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Nullary Built In Value Function Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.NullaryBuiltInValueFunctionCall#getNullaryBuiltInValueFunction <em>Nullary Built In Value Function</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getNullaryBuiltInValueFunctionCall()
 * @model
 * @generated
 */
public interface NullaryBuiltInValueFunctionCall extends BuiltInValueFunctionCall {
	/**
	 * Returns the value of the '<em><b>Nullary Built In Value Function</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.BuiltInNullaryValueFunctionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Nullary Built In Value Function</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Nullary Built In Value Function</em>' attribute.
	 * @see nqc.BuiltInNullaryValueFunctionEnum
	 * @see #setNullaryBuiltInValueFunction(BuiltInNullaryValueFunctionEnum)
	 * @see nqc.NqcPackage#getNullaryBuiltInValueFunctionCall_NullaryBuiltInValueFunction()
	 * @model required="true"
	 * @generated
	 */
	BuiltInNullaryValueFunctionEnum getNullaryBuiltInValueFunction();

	/**
	 * Sets the value of the '{@link nqc.NullaryBuiltInValueFunctionCall#getNullaryBuiltInValueFunction <em>Nullary Built In Value Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Nullary Built In Value Function</em>' attribute.
	 * @see nqc.BuiltInNullaryValueFunctionEnum
	 * @see #getNullaryBuiltInValueFunction()
	 * @generated
	 */
	void setNullaryBuiltInValueFunction(BuiltInNullaryValueFunctionEnum value);

} // NullaryBuiltInValueFunctionCall
