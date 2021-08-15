/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>chanpoll expr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.chanpoll_expr#getChan_poll <em>Chan poll</em>}</li>
 *   <li>{@link promela.chanpoll_expr#getVarref <em>Varref</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getchanpoll_expr()
 * @model
 * @generated
 */
public interface chanpoll_expr extends expr {
	/**
	 * Returns the value of the '<em><b>Chan poll</b></em>' attribute.
	 * The literals are from the enumeration {@link promela.chanpoll}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Chan poll</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Chan poll</em>' attribute.
	 * @see promela.chanpoll
	 * @see #setChan_poll(chanpoll)
	 * @see promela.PromelaPackage#getchanpoll_expr_Chan_poll()
	 * @model required="true"
	 * @generated
	 */
	chanpoll getChan_poll();

	/**
	 * Sets the value of the '{@link promela.chanpoll_expr#getChan_poll <em>Chan poll</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Chan poll</em>' attribute.
	 * @see promela.chanpoll
	 * @see #getChan_poll()
	 * @generated
	 */
	void setChan_poll(chanpoll value);

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
	 * @see promela.PromelaPackage#getchanpoll_expr_Varref()
	 * @model containment="true" required="true"
	 * @generated
	 */
	varref getVarref();

	/**
	 * Sets the value of the '{@link promela.chanpoll_expr#getVarref <em>Varref</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Varref</em>' containment reference.
	 * @see #getVarref()
	 * @generated
	 */
	void setVarref(varref value);

} // chanpoll_expr
