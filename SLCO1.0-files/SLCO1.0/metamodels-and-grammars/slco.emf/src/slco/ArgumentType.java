/**
 */
package slco;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Argument Type</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.ArgumentType#getType <em>Type</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getArgumentType()
 * @model
 * @generated
 */
public interface ArgumentType extends EObject {
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
	 * @see slco.SlcoPackage#getArgumentType_Type()
	 * @model required="true"
	 * @generated
	 */
	PrimitiveTypeEnum getType();

	/**
	 * Sets the value of the '{@link slco.ArgumentType#getType <em>Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Type</em>' attribute.
	 * @see slco.PrimitiveTypeEnum
	 * @see #getType()
	 * @generated
	 */
	void setType(PrimitiveTypeEnum value);

} // ArgumentType
