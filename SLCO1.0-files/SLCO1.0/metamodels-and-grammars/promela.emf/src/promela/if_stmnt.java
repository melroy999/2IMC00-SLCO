/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>if stmnt</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.if_stmnt#getOptions <em>Options</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getif_stmnt()
 * @model
 * @generated
 */
public interface if_stmnt extends stmnt {
	/**
	 * Returns the value of the '<em><b>Options</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Options</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Options</em>' containment reference.
	 * @see #setOptions(options)
	 * @see promela.PromelaPackage#getif_stmnt_Options()
	 * @model containment="true" required="true"
	 * @generated
	 */
	options getOptions();

	/**
	 * Sets the value of the '{@link promela.if_stmnt#getOptions <em>Options</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Options</em>' containment reference.
	 * @see #getOptions()
	 * @generated
	 */
	void setOptions(options value);

} // if_stmnt
