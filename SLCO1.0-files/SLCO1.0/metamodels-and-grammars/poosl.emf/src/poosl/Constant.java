/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.Constant#getInteger <em>Integer</em>}</li>
 *   <li>{@link poosl.Constant#getString <em>String</em>}</li>
 *   <li>{@link poosl.Constant#isBoolean <em>Boolean</em>}</li>
 *   <li>{@link poosl.Constant#getReal <em>Real</em>}</li>
 *   <li>{@link poosl.Constant#getType <em>Type</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getConstant()
 * @model
 * @generated
 */
public interface Constant extends EObject {
	/**
	 * Returns the value of the '<em><b>Integer</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Integer</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Integer</em>' attribute.
	 * @see #setInteger(int)
	 * @see poosl.PooslPackage#getConstant_Integer()
	 * @model
	 * @generated
	 */
	int getInteger();

	/**
	 * Sets the value of the '{@link poosl.Constant#getInteger <em>Integer</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Integer</em>' attribute.
	 * @see #getInteger()
	 * @generated
	 */
	void setInteger(int value);

	/**
	 * Returns the value of the '<em><b>String</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>String</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>String</em>' attribute.
	 * @see #setString(String)
	 * @see poosl.PooslPackage#getConstant_String()
	 * @model
	 * @generated
	 */
	String getString();

	/**
	 * Sets the value of the '{@link poosl.Constant#getString <em>String</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>String</em>' attribute.
	 * @see #getString()
	 * @generated
	 */
	void setString(String value);

	/**
	 * Returns the value of the '<em><b>Boolean</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Boolean</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Boolean</em>' attribute.
	 * @see #setBoolean(boolean)
	 * @see poosl.PooslPackage#getConstant_Boolean()
	 * @model
	 * @generated
	 */
	boolean isBoolean();

	/**
	 * Sets the value of the '{@link poosl.Constant#isBoolean <em>Boolean</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Boolean</em>' attribute.
	 * @see #isBoolean()
	 * @generated
	 */
	void setBoolean(boolean value);

	/**
	 * Returns the value of the '<em><b>Real</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Real</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Real</em>' attribute.
	 * @see #setReal(float)
	 * @see poosl.PooslPackage#getConstant_Real()
	 * @model
	 * @generated
	 */
	float getReal();

	/**
	 * Sets the value of the '{@link poosl.Constant#getReal <em>Real</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Real</em>' attribute.
	 * @see #getReal()
	 * @generated
	 */
	void setReal(float value);

	/**
	 * Returns the value of the '<em><b>Type</b></em>' attribute.
	 * The literals are from the enumeration {@link poosl.PrimitiveTypeEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Type</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Type</em>' attribute.
	 * @see poosl.PrimitiveTypeEnum
	 * @see #setType(PrimitiveTypeEnum)
	 * @see poosl.PooslPackage#getConstant_Type()
	 * @model
	 * @generated
	 */
	PrimitiveTypeEnum getType();

	/**
	 * Sets the value of the '{@link poosl.Constant#getType <em>Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Type</em>' attribute.
	 * @see poosl.PrimitiveTypeEnum
	 * @see #getType()
	 * @generated
	 */
	void setType(PrimitiveTypeEnum value);

} // Constant
