/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Primitive Type</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.PrimitiveType#getPrimitiveTypeEnum <em>Primitive Type Enum</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getPrimitiveType()
 * @model
 * @generated
 */
public interface PrimitiveType extends Type {
	/**
	 * Returns the value of the '<em><b>Primitive Type Enum</b></em>' attribute.
	 * The literals are from the enumeration {@link poosl.PrimitiveTypeEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Primitive Type Enum</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Primitive Type Enum</em>' attribute.
	 * @see poosl.PrimitiveTypeEnum
	 * @see #setPrimitiveTypeEnum(PrimitiveTypeEnum)
	 * @see poosl.PooslPackage#getPrimitiveType_PrimitiveTypeEnum()
	 * @model
	 * @generated
	 */
	PrimitiveTypeEnum getPrimitiveTypeEnum();

	/**
	 * Sets the value of the '{@link poosl.PrimitiveType#getPrimitiveTypeEnum <em>Primitive Type Enum</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Primitive Type Enum</em>' attribute.
	 * @see poosl.PrimitiveTypeEnum
	 * @see #getPrimitiveTypeEnum()
	 * @generated
	 */
	void setPrimitiveTypeEnum(PrimitiveTypeEnum value);

} // PrimitiveType
