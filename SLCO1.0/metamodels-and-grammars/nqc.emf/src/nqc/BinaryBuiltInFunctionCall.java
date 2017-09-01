/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Binary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.BinaryBuiltInFunctionCall#getBinaryBuiltInFunction <em>Binary Built In Function</em>}</li>
 *   <li>{@link nqc.BinaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}</li>
 *   <li>{@link nqc.BinaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getBinaryBuiltInFunctionCall()
 * @model
 * @generated
 */
public interface BinaryBuiltInFunctionCall extends BuiltInFunctionCall {
	/**
	 * Returns the value of the '<em><b>Binary Built In Function</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.BuiltInBinaryFunctionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Binary Built In Function</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Binary Built In Function</em>' attribute.
	 * @see nqc.BuiltInBinaryFunctionEnum
	 * @see #setBinaryBuiltInFunction(BuiltInBinaryFunctionEnum)
	 * @see nqc.NqcPackage#getBinaryBuiltInFunctionCall_BinaryBuiltInFunction()
	 * @model required="true"
	 * @generated
	 */
	BuiltInBinaryFunctionEnum getBinaryBuiltInFunction();

	/**
	 * Sets the value of the '{@link nqc.BinaryBuiltInFunctionCall#getBinaryBuiltInFunction <em>Binary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Binary Built In Function</em>' attribute.
	 * @see nqc.BuiltInBinaryFunctionEnum
	 * @see #getBinaryBuiltInFunction()
	 * @generated
	 */
	void setBinaryBuiltInFunction(BuiltInBinaryFunctionEnum value);

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
	 * @see nqc.NqcPackage#getBinaryBuiltInFunctionCall_Parameter1()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter1();

	/**
	 * Sets the value of the '{@link nqc.BinaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}' containment reference.
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
	 * @see nqc.NqcPackage#getBinaryBuiltInFunctionCall_Parameter2()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter2();

	/**
	 * Sets the value of the '{@link nqc.BinaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter2</em>' containment reference.
	 * @see #getParameter2()
	 * @generated
	 */
	void setParameter2(Expression value);

} // BinaryBuiltInFunctionCall
