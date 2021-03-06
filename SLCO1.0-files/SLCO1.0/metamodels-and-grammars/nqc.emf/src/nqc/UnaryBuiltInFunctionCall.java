/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Unary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.UnaryBuiltInFunctionCall#getUnaryBuiltInFunction <em>Unary Built In Function</em>}</li>
 *   <li>{@link nqc.UnaryBuiltInFunctionCall#getParameter <em>Parameter</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getUnaryBuiltInFunctionCall()
 * @model
 * @generated
 */
public interface UnaryBuiltInFunctionCall extends BuiltInFunctionCall {
	/**
	 * Returns the value of the '<em><b>Unary Built In Function</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.BuiltInUnaryFunctionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Unary Built In Function</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Unary Built In Function</em>' attribute.
	 * @see nqc.BuiltInUnaryFunctionEnum
	 * @see #setUnaryBuiltInFunction(BuiltInUnaryFunctionEnum)
	 * @see nqc.NqcPackage#getUnaryBuiltInFunctionCall_UnaryBuiltInFunction()
	 * @model required="true"
	 * @generated
	 */
	BuiltInUnaryFunctionEnum getUnaryBuiltInFunction();

	/**
	 * Sets the value of the '{@link nqc.UnaryBuiltInFunctionCall#getUnaryBuiltInFunction <em>Unary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Unary Built In Function</em>' attribute.
	 * @see nqc.BuiltInUnaryFunctionEnum
	 * @see #getUnaryBuiltInFunction()
	 * @generated
	 */
	void setUnaryBuiltInFunction(BuiltInUnaryFunctionEnum value);

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
	 * @see nqc.NqcPackage#getUnaryBuiltInFunctionCall_Parameter()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter();

	/**
	 * Sets the value of the '{@link nqc.UnaryBuiltInFunctionCall#getParameter <em>Parameter</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter</em>' containment reference.
	 * @see #getParameter()
	 * @generated
	 */
	void setParameter(Expression value);

} // UnaryBuiltInFunctionCall
