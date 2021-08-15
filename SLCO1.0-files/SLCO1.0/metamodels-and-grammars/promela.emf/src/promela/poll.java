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
 * A representation of the model object '<em><b>poll</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.poll#getVarref <em>Varref</em>}</li>
 *   <li>{@link promela.poll#getRecv_args <em>Recv args</em>}</li>
 *   <li>{@link promela.poll#getPolltype <em>Polltype</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getpoll()
 * @model abstract="true"
 * @generated
 */
public interface poll extends any_expr {
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
	 * @see promela.PromelaPackage#getpoll_Varref()
	 * @model containment="true" required="true"
	 * @generated
	 */
	varref getVarref();

	/**
	 * Sets the value of the '{@link promela.poll#getVarref <em>Varref</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Varref</em>' containment reference.
	 * @see #getVarref()
	 * @generated
	 */
	void setVarref(varref value);

	/**
	 * Returns the value of the '<em><b>Recv args</b></em>' containment reference list.
	 * The list contents are of type {@link promela.recv_arg}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Recv args</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Recv args</em>' containment reference list.
	 * @see promela.PromelaPackage#getpoll_Recv_args()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<recv_arg> getRecv_args();

	/**
	 * Returns the value of the '<em><b>Polltype</b></em>' attribute.
	 * The default value is <code>"0"</code>.
	 * The literals are from the enumeration {@link promela.polltypeenum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Polltype</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Polltype</em>' attribute.
	 * @see promela.polltypeenum
	 * @see #setPolltype(polltypeenum)
	 * @see promela.PromelaPackage#getpoll_Polltype()
	 * @model default="0" required="true"
	 * @generated
	 */
	polltypeenum getPolltype();

	/**
	 * Sets the value of the '{@link promela.poll#getPolltype <em>Polltype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Polltype</em>' attribute.
	 * @see promela.polltypeenum
	 * @see #getPolltype()
	 * @generated
	 */
	void setPolltype(polltypeenum value);

} // poll
