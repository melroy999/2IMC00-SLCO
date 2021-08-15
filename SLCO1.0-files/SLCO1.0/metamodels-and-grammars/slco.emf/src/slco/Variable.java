/**
 */
package slco;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Variable</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.Variable#getName <em>Name</em>}</li>
 *   <li>{@link slco.Variable#getInitialValue <em>Initial Value</em>}</li>
 *   <li>{@link slco.Variable#getType <em>Type</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getVariable()
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
	 * @see slco.SlcoPackage#getVariable_Name()
	 * @model required="true"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link slco.Variable#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Initial Value</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Initial Value</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Initial Value</em>' containment reference.
	 * @see #setInitialValue(Expression)
	 * @see slco.SlcoPackage#getVariable_InitialValue()
	 * @model containment="true"
	 * @generated
	 */
	Expression getInitialValue();

	/**
	 * Sets the value of the '{@link slco.Variable#getInitialValue <em>Initial Value</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Initial Value</em>' containment reference.
	 * @see #getInitialValue()
	 * @generated
	 */
	void setInitialValue(Expression value);

	/**
	 * Returns the value of the '<em><b>Type</b></em>' attribute.
	 * The literals are from the enumeration {@link slco.PrimitiveTypeEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Type</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Type</em>' attribute.
	 * @see slco.PrimitiveTypeEnum
	 * @see #setType(PrimitiveTypeEnum)
	 * @see slco.SlcoPackage#getVariable_Type()
	 * @model required="true"
	 * @generated
	 */
	PrimitiveTypeEnum getType();

	/**
	 * Sets the value of the '{@link slco.Variable#getType <em>Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Type</em>' attribute.
	 * @see slco.PrimitiveTypeEnum
	 * @see #getType()
	 * @generated
	 */
	void setType(PrimitiveTypeEnum value);

} // Variable
