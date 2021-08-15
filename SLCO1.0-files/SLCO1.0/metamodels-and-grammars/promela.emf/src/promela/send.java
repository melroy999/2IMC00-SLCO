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
 * A representation of the model object '<em><b>send</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.send#getVarref <em>Varref</em>}</li>
 *   <li>{@link promela.send#getSend_args <em>Send args</em>}</li>
 *   <li>{@link promela.send#getSendtype <em>Sendtype</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getsend()
 * @model
 * @generated
 */
public interface send extends stmnt {
	/**
	 * Returns the value of the '<em><b>Varref</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Varref</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Varref</em>' containment reference.
	 * @see #setVarref(varref)
	 * @see promela.PromelaPackage#getsend_Varref()
	 * @model containment="true" required="true"
	 * @generated
	 */
	varref getVarref();

	/**
	 * Sets the value of the '{@link promela.send#getVarref <em>Varref</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Varref</em>' containment reference.
	 * @see #getVarref()
	 * @generated
	 */
	void setVarref(varref value);

	/**
	 * Returns the value of the '<em><b>Send args</b></em>' containment reference list.
	 * The list contents are of type {@link promela.any_expr}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Send args</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Send args</em>' containment reference list.
	 * @see promela.PromelaPackage#getsend_Send_args()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<any_expr> getSend_args();

	/**
	 * Returns the value of the '<em><b>Sendtype</b></em>' attribute.
	 * The default value is <code>"0"</code>.
	 * The literals are from the enumeration {@link promela.sendtypeenum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sendtype</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sendtype</em>' attribute.
	 * @see promela.sendtypeenum
	 * @see #setSendtype(sendtypeenum)
	 * @see promela.PromelaPackage#getsend_Sendtype()
	 * @model default="0" required="true"
	 * @generated
	 */
	sendtypeenum getSendtype();

	/**
	 * Sets the value of the '{@link promela.send#getSendtype <em>Sendtype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sendtype</em>' attribute.
	 * @see promela.sendtypeenum
	 * @see #getSendtype()
	 * @generated
	 */
	void setSendtype(sendtypeenum value);

} // send
