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
 * A representation of the model object '<em><b>receive</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.receive#getVarref <em>Varref</em>}</li>
 *   <li>{@link promela.receive#getRecv_args <em>Recv args</em>}</li>
 *   <li>{@link promela.receive#getReceivetype <em>Receivetype</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getreceive()
 * @model
 * @generated
 */
public interface receive extends stmnt {
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
	 * @see promela.PromelaPackage#getreceive_Varref()
	 * @model containment="true" required="true"
	 * @generated
	 */
	varref getVarref();

	/**
	 * Sets the value of the '{@link promela.receive#getVarref <em>Varref</em>}' containment reference.
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
	 * @see promela.PromelaPackage#getreceive_Recv_args()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<recv_arg> getRecv_args();

	/**
	 * Returns the value of the '<em><b>Receivetype</b></em>' attribute.
	 * The default value is <code>"0"</code>.
	 * The literals are from the enumeration {@link promela.receivetypeenum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Receivetype</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Receivetype</em>' attribute.
	 * @see promela.receivetypeenum
	 * @see #setReceivetype(receivetypeenum)
	 * @see promela.PromelaPackage#getreceive_Receivetype()
	 * @model default="0" required="true"
	 * @generated
	 */
	receivetypeenum getReceivetype();

	/**
	 * Sets the value of the '{@link promela.receive#getReceivetype <em>Receivetype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Receivetype</em>' attribute.
	 * @see promela.receivetypeenum
	 * @see #getReceivetype()
	 * @generated
	 */
	void setReceivetype(receivetypeenum value);

} // receive
