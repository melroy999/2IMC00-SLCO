/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>stmnt</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.stmnt#getLabel <em>Label</em>}</li>
 *   <li>{@link promela.stmnt#getUnless <em>Unless</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getstmnt()
 * @model abstract="true"
 * @generated
 */
public interface stmnt extends step {
	/**
	 * Returns the value of the '<em><b>Label</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Label</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Label</em>' attribute.
	 * @see #setLabel(String)
	 * @see promela.PromelaPackage#getstmnt_Label()
	 * @model
	 * @generated
	 */
	String getLabel();

	/**
	 * Sets the value of the '{@link promela.stmnt#getLabel <em>Label</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Label</em>' attribute.
	 * @see #getLabel()
	 * @generated
	 */
	void setLabel(String value);

	/**
	 * Returns the value of the '<em><b>Unless</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Unless</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Unless</em>' containment reference.
	 * @see #setUnless(stmnt)
	 * @see promela.PromelaPackage#getstmnt_Unless()
	 * @model containment="true"
	 * @generated
	 */
	stmnt getUnless();

	/**
	 * Sets the value of the '{@link promela.stmnt#getUnless <em>Unless</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Unless</em>' containment reference.
	 * @see #getUnless()
	 * @generated
	 */
	void setUnless(stmnt value);

} // stmnt
