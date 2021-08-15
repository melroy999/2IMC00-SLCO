/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>cexpr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.c_expr#getEmbedded_c_code <em>Embedded ccode</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getc_expr()
 * @model
 * @generated
 */
public interface c_expr extends stmnt {
	/**
	 * Returns the value of the '<em><b>Embedded ccode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Embedded ccode</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Embedded ccode</em>' attribute.
	 * @see #setEmbedded_c_code(String)
	 * @see promela.PromelaPackage#getc_expr_Embedded_c_code()
	 * @model required="true"
	 * @generated
	 */
	String getEmbedded_c_code();

	/**
	 * Sets the value of the '{@link promela.c_expr#getEmbedded_c_code <em>Embedded ccode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Embedded ccode</em>' attribute.
	 * @see #getEmbedded_c_code()
	 * @generated
	 */
	void setEmbedded_c_code(String value);

} // c_expr
