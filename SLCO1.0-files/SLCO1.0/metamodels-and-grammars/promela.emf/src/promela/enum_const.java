/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>enum const</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.enum_const#getConst_enum <em>Const enum</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getenum_const()
 * @model
 * @generated
 */
public interface enum_const extends constant {
	/**
	 * Returns the value of the '<em><b>Const enum</b></em>' attribute.
	 * The literals are from the enumeration {@link promela.constenum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Const enum</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Const enum</em>' attribute.
	 * @see promela.constenum
	 * @see #setConst_enum(constenum)
	 * @see promela.PromelaPackage#getenum_const_Const_enum()
	 * @model required="true"
	 * @generated
	 */
	constenum getConst_enum();

	/**
	 * Sets the value of the '{@link promela.enum_const#getConst_enum <em>Const enum</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Const enum</em>' attribute.
	 * @see promela.constenum
	 * @see #getConst_enum()
	 * @generated
	 */
	void setConst_enum(constenum value);

} // enum_const
