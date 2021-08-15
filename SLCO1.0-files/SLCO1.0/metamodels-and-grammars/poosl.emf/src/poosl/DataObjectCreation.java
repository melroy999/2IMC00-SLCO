/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Data Object Creation</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.DataObjectCreation#getDataClass <em>Data Class</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getDataObjectCreation()
 * @model
 * @generated
 */
public interface DataObjectCreation extends Expression {
	/**
	 * Returns the value of the '<em><b>Data Class</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Data Class</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Data Class</em>' reference.
	 * @see #setDataClass(DataClass)
	 * @see poosl.PooslPackage#getDataObjectCreation_DataClass()
	 * @model required="true"
	 * @generated
	 */
	DataClass getDataClass();

	/**
	 * Sets the value of the '{@link poosl.DataObjectCreation#getDataClass <em>Data Class</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Data Class</em>' reference.
	 * @see #getDataClass()
	 * @generated
	 */
	void setDataClass(DataClass value);

} // DataObjectCreation
