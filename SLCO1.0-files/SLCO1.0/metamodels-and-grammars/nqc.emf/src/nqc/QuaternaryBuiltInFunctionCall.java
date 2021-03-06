/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Quaternary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.QuaternaryBuiltInFunctionCall#getQuaternaryBuiltInFunction <em>Quaternary Built In Function</em>}</li>
 *   <li>{@link nqc.QuaternaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}</li>
 *   <li>{@link nqc.QuaternaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}</li>
 *   <li>{@link nqc.QuaternaryBuiltInFunctionCall#getParameter3 <em>Parameter3</em>}</li>
 *   <li>{@link nqc.QuaternaryBuiltInFunctionCall#getParameter4 <em>Parameter4</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getQuaternaryBuiltInFunctionCall()
 * @model
 * @generated
 */
public interface QuaternaryBuiltInFunctionCall extends BuiltInFunctionCall {
	/**
	 * Returns the value of the '<em><b>Quaternary Built In Function</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.BuiltInQuaternaryFunctionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Quaternary Built In Function</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Quaternary Built In Function</em>' attribute.
	 * @see nqc.BuiltInQuaternaryFunctionEnum
	 * @see #setQuaternaryBuiltInFunction(BuiltInQuaternaryFunctionEnum)
	 * @see nqc.NqcPackage#getQuaternaryBuiltInFunctionCall_QuaternaryBuiltInFunction()
	 * @model required="true"
	 * @generated
	 */
	BuiltInQuaternaryFunctionEnum getQuaternaryBuiltInFunction();

	/**
	 * Sets the value of the '{@link nqc.QuaternaryBuiltInFunctionCall#getQuaternaryBuiltInFunction <em>Quaternary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Quaternary Built In Function</em>' attribute.
	 * @see nqc.BuiltInQuaternaryFunctionEnum
	 * @see #getQuaternaryBuiltInFunction()
	 * @generated
	 */
	void setQuaternaryBuiltInFunction(BuiltInQuaternaryFunctionEnum value);

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
	 * @see nqc.NqcPackage#getQuaternaryBuiltInFunctionCall_Parameter1()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter1();

	/**
	 * Sets the value of the '{@link nqc.QuaternaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}' containment reference.
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
	 * @see nqc.NqcPackage#getQuaternaryBuiltInFunctionCall_Parameter2()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter2();

	/**
	 * Sets the value of the '{@link nqc.QuaternaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}' containment reference.
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
	 * @see nqc.NqcPackage#getQuaternaryBuiltInFunctionCall_Parameter3()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter3();

	/**
	 * Sets the value of the '{@link nqc.QuaternaryBuiltInFunctionCall#getParameter3 <em>Parameter3</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter3</em>' containment reference.
	 * @see #getParameter3()
	 * @generated
	 */
	void setParameter3(Expression value);

	/**
	 * Returns the value of the '<em><b>Parameter4</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter4</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter4</em>' containment reference.
	 * @see #setParameter4(Expression)
	 * @see nqc.NqcPackage#getQuaternaryBuiltInFunctionCall_Parameter4()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter4();

	/**
	 * Sets the value of the '{@link nqc.QuaternaryBuiltInFunctionCall#getParameter4 <em>Parameter4</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter4</em>' containment reference.
	 * @see #getParameter4()
	 * @generated
	 */
	void setParameter4(Expression value);

} // QuaternaryBuiltInFunctionCall
