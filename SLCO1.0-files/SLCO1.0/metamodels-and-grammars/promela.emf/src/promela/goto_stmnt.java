/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>goto stmnt</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.goto_stmnt#getStmnt <em>Stmnt</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getgoto_stmnt()
 * @model
 * @generated
 */
public interface goto_stmnt extends stmnt {
	/**
	 * Returns the value of the '<em><b>Stmnt</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Stmnt</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Stmnt</em>' reference.
	 * @see #setStmnt(stmnt)
	 * @see promela.PromelaPackage#getgoto_stmnt_Stmnt()
	 * @model required="true"
	 * @generated
	 */
	stmnt getStmnt();

	/**
	 * Sets the value of the '{@link promela.goto_stmnt#getStmnt <em>Stmnt</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Stmnt</em>' reference.
	 * @see #getStmnt()
	 * @generated
	 */
	void setStmnt(stmnt value);

} // goto_stmnt
