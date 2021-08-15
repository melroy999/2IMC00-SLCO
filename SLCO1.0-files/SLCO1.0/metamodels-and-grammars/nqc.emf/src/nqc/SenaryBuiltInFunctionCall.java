/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Senary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.SenaryBuiltInFunctionCall#getSenaryBuiltInFunction <em>Senary Built In Function</em>}</li>
 *   <li>{@link nqc.SenaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}</li>
 *   <li>{@link nqc.SenaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}</li>
 *   <li>{@link nqc.SenaryBuiltInFunctionCall#getParameter3 <em>Parameter3</em>}</li>
 *   <li>{@link nqc.SenaryBuiltInFunctionCall#getParameter4 <em>Parameter4</em>}</li>
 *   <li>{@link nqc.SenaryBuiltInFunctionCall#getParameter5 <em>Parameter5</em>}</li>
 *   <li>{@link nqc.SenaryBuiltInFunctionCall#getParameter6 <em>Parameter6</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getSenaryBuiltInFunctionCall()
 * @model
 * @generated
 */
public interface SenaryBuiltInFunctionCall extends BuiltInFunctionCall {
	/**
	 * Returns the value of the '<em><b>Senary Built In Function</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.BuiltInSenaryFunctionEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Senary Built In Function</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Senary Built In Function</em>' attribute.
	 * @see nqc.BuiltInSenaryFunctionEnum
	 * @see #setSenaryBuiltInFunction(BuiltInSenaryFunctionEnum)
	 * @see nqc.NqcPackage#getSenaryBuiltInFunctionCall_SenaryBuiltInFunction()
	 * @model required="true"
	 * @generated
	 */
	BuiltInSenaryFunctionEnum getSenaryBuiltInFunction();

	/**
	 * Sets the value of the '{@link nqc.SenaryBuiltInFunctionCall#getSenaryBuiltInFunction <em>Senary Built In Function</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Senary Built In Function</em>' attribute.
	 * @see nqc.BuiltInSenaryFunctionEnum
	 * @see #getSenaryBuiltInFunction()
	 * @generated
	 */
	void setSenaryBuiltInFunction(BuiltInSenaryFunctionEnum value);

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
	 * @see nqc.NqcPackage#getSenaryBuiltInFunctionCall_Parameter1()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter1();

	/**
	 * Sets the value of the '{@link nqc.SenaryBuiltInFunctionCall#getParameter1 <em>Parameter1</em>}' containment reference.
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
	 * @see nqc.NqcPackage#getSenaryBuiltInFunctionCall_Parameter2()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter2();

	/**
	 * Sets the value of the '{@link nqc.SenaryBuiltInFunctionCall#getParameter2 <em>Parameter2</em>}' containment reference.
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
	 * @see nqc.NqcPackage#getSenaryBuiltInFunctionCall_Parameter3()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter3();

	/**
	 * Sets the value of the '{@link nqc.SenaryBuiltInFunctionCall#getParameter3 <em>Parameter3</em>}' containment reference.
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
	 * @see nqc.NqcPackage#getSenaryBuiltInFunctionCall_Parameter4()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter4();

	/**
	 * Sets the value of the '{@link nqc.SenaryBuiltInFunctionCall#getParameter4 <em>Parameter4</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter4</em>' containment reference.
	 * @see #getParameter4()
	 * @generated
	 */
	void setParameter4(Expression value);

	/**
	 * Returns the value of the '<em><b>Parameter5</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter5</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter5</em>' containment reference.
	 * @see #setParameter5(Expression)
	 * @see nqc.NqcPackage#getSenaryBuiltInFunctionCall_Parameter5()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter5();

	/**
	 * Sets the value of the '{@link nqc.SenaryBuiltInFunctionCall#getParameter5 <em>Parameter5</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter5</em>' containment reference.
	 * @see #getParameter5()
	 * @generated
	 */
	void setParameter5(Expression value);

	/**
	 * Returns the value of the '<em><b>Parameter6</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter6</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter6</em>' containment reference.
	 * @see #setParameter6(Expression)
	 * @see nqc.NqcPackage#getSenaryBuiltInFunctionCall_Parameter6()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Expression getParameter6();

	/**
	 * Sets the value of the '{@link nqc.SenaryBuiltInFunctionCall#getParameter6 <em>Parameter6</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter6</em>' containment reference.
	 * @see #getParameter6()
	 * @generated
	 */
	void setParameter6(Expression value);

} // SenaryBuiltInFunctionCall
