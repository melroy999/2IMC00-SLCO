/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Binary Built In Value Function Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.BinaryBuiltInValueFunctionCall#getBinaryBuiltInValueFunction <em>Binary Built In Value Function</em>}</li>
 *   <li>{@link nqc.BinaryBuiltInValueFunctionCall#getParameter1 <em>Parameter1</em>}</li>
 *   <li>{@link nqc.BinaryBuiltInValueFunctionCall#getParameter2 <em>Parameter2</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getBinaryBuiltInValueFunctionCall()
 * @model
 * @generated
 */
public interface BinaryBuiltInValueFunctionCall extends BuiltInValueFunctionCall {
	/**
	 * Returns the value of the '<em><b>Binary Built In Value Function</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.BuiltInBinaryValueFunctionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Binary Built In Value Function</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Binary Built In Value Function</em>' attribute.
	 * @see nqc.BuiltInBinaryValueFunctionEnum
	 * @see #setBinaryBuiltInValueFunction(BuiltInBinaryValueFunctionEnum)
	 * @see nqc.NqcPackage#getBinaryBuiltInValueFunctionCall_BinaryBuiltInValueFunction()
	 * @model required="true"
	 * @generated
	 */
	BuiltInBinaryValueFunctionEnum getBinaryBuiltInValueFunction();

	/**
	 * Sets the value of the '{@link nqc.BinaryBuiltInValueFunctionCall#getBinaryBuiltInValueFunction <em>Binary Built In Value Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Binary Built In Value Function</em>' attribute.
	 * @see nqc.BuiltInBinaryValueFunctionEnum
	 * @see #getBinaryBuiltInValueFunction()
	 * @generated
	 */
	void setBinaryBuiltInValueFunction(BuiltInBinaryValueFunctionEnum value);

	/**
	 * Returns the value of the '<em><b>Parameter1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter1</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter1</em>' containment reference.
	 * @see #setParameter1(Expression)
	 * @see nqc.NqcPackage#getBinaryBuiltInValueFunctionCall_Parameter1()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter1();

	/**
	 * Sets the value of the '{@link nqc.BinaryBuiltInValueFunctionCall#getParameter1 <em>Parameter1</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter1</em>' containment reference.
	 * @see #getParameter1()
	 * @generated
	 */
	void setParameter1(Expression value);

	/**
	 * Returns the value of the '<em><b>Parameter2</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter2</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter2</em>' containment reference.
	 * @see #setParameter2(Expression)
	 * @see nqc.NqcPackage#getBinaryBuiltInValueFunctionCall_Parameter2()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter2();

	/**
	 * Sets the value of the '{@link nqc.BinaryBuiltInValueFunctionCall#getParameter2 <em>Parameter2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter2</em>' containment reference.
	 * @see #getParameter2()
	 * @generated
	 */
	void setParameter2(Expression value);

} // BinaryBuiltInValueFunctionCall
