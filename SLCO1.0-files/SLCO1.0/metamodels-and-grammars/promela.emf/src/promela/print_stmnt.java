/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>print stmnt</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.print_stmnt#getString <em>String</em>}</li>
 *   <li>{@link promela.print_stmnt#getArg_lst <em>Arg lst</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getprint_stmnt()
 * @model
 * @generated
 */
public interface print_stmnt extends stmnt {
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
	 * @see promela.PromelaPackage#getprint_stmnt_String()
	 * @model required="true"
	 * @generated
	 */
	String getString();

	/**
	 * Sets the value of the '{@link promela.print_stmnt#getString <em>String</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>String</em>' attribute.
	 * @see #getString()
	 * @generated
	 */
	void setString(String value);

	/**
	 * Returns the value of the '<em><b>Arg lst</b></em>' containment reference list.
	 * The list contents are of type {@link promela.any_expr}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Arg lst</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Arg lst</em>' containment reference list.
	 * @see promela.PromelaPackage#getprint_stmnt_Arg_lst()
	 * @model containment="true"
	 * @generated
	 */
	EList<any_expr> getArg_lst();

} // print_stmnt
