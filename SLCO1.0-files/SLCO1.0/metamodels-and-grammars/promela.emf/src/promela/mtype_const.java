/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>mtype const</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.mtype_const#getMtype <em>Mtype</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getmtype_const()
 * @model
 * @generated
 */
public interface mtype_const extends constant, recv_arg {
	/**
	 * Returns the value of the '<em><b>Mtype</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Mtype</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Mtype</em>' reference.
	 * @see #setMtype(mtype)
	 * @see promela.PromelaPackage#getmtype_const_Mtype()
	 * @model required="true"
	 * @generated
	 */
	mtype getMtype();

	/**
	 * Sets the value of the '{@link promela.mtype_const#getMtype <em>Mtype</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Mtype</em>' reference.
	 * @see #getMtype()
	 * @generated
	 */
	void setMtype(mtype value);

} // mtype_const
