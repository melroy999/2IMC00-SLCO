/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Unary Built In Value Function Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.UnaryBuiltInValueFunctionCall#getUnaryBuiltInValueFunction <em>Unary Built In Value Function</em>}</li>
 *   <li>{@link nqc.UnaryBuiltInValueFunctionCall#getParameter <em>Parameter</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getUnaryBuiltInValueFunctionCall()
 * @model
 * @generated
 */
public interface UnaryBuiltInValueFunctionCall extends BuiltInValueFunctionCall {
	/**
	 * Returns the value of the '<em><b>Unary Built In Value Function</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.BuiltInUnaryValueFunctionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Unary Built In Value Function</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Unary Built In Value Function</em>' attribute.
	 * @see nqc.BuiltInUnaryValueFunctionEnum
	 * @see #setUnaryBuiltInValueFunction(BuiltInUnaryValueFunctionEnum)
	 * @see nqc.NqcPackage#getUnaryBuiltInValueFunctionCall_UnaryBuiltInValueFunction()
	 * @model required="true"
	 * @generated
	 */
	BuiltInUnaryValueFunctionEnum getUnaryBuiltInValueFunction();

	/**
	 * Sets the value of the '{@link nqc.UnaryBuiltInValueFunctionCall#getUnaryBuiltInValueFunction <em>Unary Built In Value Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Unary Built In Value Function</em>' attribute.
	 * @see nqc.BuiltInUnaryValueFunctionEnum
	 * @see #getUnaryBuiltInValueFunction()
	 * @generated
	 */
	void setUnaryBuiltInValueFunction(BuiltInUnaryValueFunctionEnum value);

	/**
	 * Returns the value of the '<em><b>Parameter</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter</em>' containment reference.
	 * @see #setParameter(Expression)
	 * @see nqc.NqcPackage#getUnaryBuiltInValueFunctionCall_Parameter()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter();

	/**
	 * Sets the value of the '{@link nqc.UnaryBuiltInValueFunctionCall#getParameter <em>Parameter</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter</em>' containment reference.
	 * @see #getParameter()
	 * @generated
	 */
	void setParameter(Expression value);

} // UnaryBuiltInValueFunctionCall
