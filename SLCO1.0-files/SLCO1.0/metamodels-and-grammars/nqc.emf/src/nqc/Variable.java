/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Variable</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.Variable#getName <em>Name</em>}</li>
 *   <li>{@link nqc.Variable#getType <em>Type</em>}</li>
 *   <li>{@link nqc.Variable#getInitialValue <em>Initial Value</em>}</li>
 *   <li>{@link nqc.Variable#getArrayMaxSize <em>Array Max Size</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getVariable()
 * @model
 * @generated
 */
public interface Variable extends EObject {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see nqc.NqcPackage#getVariable_Name()
	 * @model unique="false" required="true" ordered="false"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link nqc.Variable#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Type</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.TypeEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Type</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Type</em>' attribute.
	 * @see nqc.TypeEnum
	 * @see #setType(TypeEnum)
	 * @see nqc.NqcPackage#getVariable_Type()
	 * @model
	 * @generated
	 */
	TypeEnum getType();

	/**
	 * Sets the value of the '{@link nqc.Variable#getType <em>Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Type</em>' attribute.
	 * @see nqc.TypeEnum
	 * @see #getType()
	 * @generated
	 */
	void setType(TypeEnum value);

	/**
	 * Returns the value of the '<em><b>Initial Value</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Initial Value</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Initial Value</em>' containment reference.
	 * @see #setInitialValue(ConstantExpression)
	 * @see nqc.NqcPackage#getVariable_InitialValue()
	 * @model containment="true"
	 * @generated
	 */
	ConstantExpression getInitialValue();

	/**
	 * Sets the value of the '{@link nqc.Variable#getInitialValue <em>Initial Value</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Initial Value</em>' containment reference.
	 * @see #getInitialValue()
	 * @generated
	 */
	void setInitialValue(ConstantExpression value);

	/**
	 * Returns the value of the '<em><b>Array Max Size</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Array Max Size</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Array Max Size</em>' reference.
	 * @see #setArrayMaxSize(IntegerConstant)
	 * @see nqc.NqcPackage#getVariable_ArrayMaxSize()
	 * @model
	 * @generated
	 */
	IntegerConstant getArrayMaxSize();

	/**
	 * Sets the value of the '{@link nqc.Variable#getArrayMaxSize <em>Array Max Size</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Array Max Size</em>' reference.
	 * @see #getArrayMaxSize()
	 * @generated
	 */
	void setArrayMaxSize(IntegerConstant value);

} // Variable
