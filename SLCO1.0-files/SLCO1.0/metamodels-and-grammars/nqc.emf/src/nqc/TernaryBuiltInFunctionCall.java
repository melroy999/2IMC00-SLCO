/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Ternary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.TernaryBuiltInFunctionCall#getTernaryBuiltInFunction <em>Ternary Built In Function</em>}</li>
 *   <li>{@link nqc.TernaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}</li>
 *   <li>{@link nqc.TernaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}</li>
 *   <li>{@link nqc.TernaryBuiltInFunctionCall#getParameter3 <em>Parameter3</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getTernaryBuiltInFunctionCall()
 * @model
 * @generated
 */
public interface TernaryBuiltInFunctionCall extends BuiltInFunctionCall {
	/**
	 * Returns the value of the '<em><b>Ternary Built In Function</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.BuiltInTernaryFunctionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Ternary Built In Function</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Ternary Built In Function</em>' attribute.
	 * @see nqc.BuiltInTernaryFunctionEnum
	 * @see #setTernaryBuiltInFunction(BuiltInTernaryFunctionEnum)
	 * @see nqc.NqcPackage#getTernaryBuiltInFunctionCall_TernaryBuiltInFunction()
	 * @model required="true"
	 * @generated
	 */
	BuiltInTernaryFunctionEnum getTernaryBuiltInFunction();

	/**
	 * Sets the value of the '{@link nqc.TernaryBuiltInFunctionCall#getTernaryBuiltInFunction <em>Ternary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Ternary Built In Function</em>' attribute.
	 * @see nqc.BuiltInTernaryFunctionEnum
	 * @see #getTernaryBuiltInFunction()
	 * @generated
	 */
	void setTernaryBuiltInFunction(BuiltInTernaryFunctionEnum value);

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
	 * @see nqc.NqcPackage#getTernaryBuiltInFunctionCall_Parameter1()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter1();

	/**
	 * Sets the value of the '{@link nqc.TernaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}' containment reference.
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
	 * @see nqc.NqcPackage#getTernaryBuiltInFunctionCall_Parameter2()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter2();

	/**
	 * Sets the value of the '{@link nqc.TernaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter2</em>' containment reference.
	 * @see #getParameter2()
	 * @generated
	 */
	void setParameter2(Expression value);

	/**
	 * Returns the value of the '<em><b>Parameter3</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter3</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter3</em>' containment reference.
	 * @see #setParameter3(Expression)
	 * @see nqc.NqcPackage#getTernaryBuiltInFunctionCall_Parameter3()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter3();

	/**
	 * Sets the value of the '{@link nqc.TernaryBuiltInFunctionCall#getParameter3 <em>Parameter3</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter3</em>' containment reference.
	 * @see #getParameter3()
	 * @generated
	 */
	void setParameter3(Expression value);

} // TernaryBuiltInFunctionCall
