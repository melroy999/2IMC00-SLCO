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
 * A representation of the model object '<em><b>one decl</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.one_decl#getVisible <em>Visible</em>}</li>
 *   <li>{@link promela.one_decl#getTypename <em>Typename</em>}</li>
 *   <li>{@link promela.one_decl#getIvars <em>Ivars</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getone_decl()
 * @model
 * @generated
 */
public interface one_decl extends step {
	/**
	 * Returns the value of the '<em><b>Visible</b></em>' attribute.
	 * The literals are from the enumeration {@link promela.visibleenum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Visible</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Visible</em>' attribute.
	 * @see promela.visibleenum
	 * @see #setVisible(visibleenum)
	 * @see promela.PromelaPackage#getone_decl_Visible()
	 * @model
	 * @generated
	 */
	visibleenum getVisible();

	/**
	 * Sets the value of the '{@link promela.one_decl#getVisible <em>Visible</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Visible</em>' attribute.
	 * @see promela.visibleenum
	 * @see #getVisible()
	 * @generated
	 */
	void setVisible(visibleenum value);

	/**
	 * Returns the value of the '<em><b>Typename</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Typename</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Typename</em>' reference.
	 * @see #setTypename(typename)
	 * @see promela.PromelaPackage#getone_decl_Typename()
	 * @model required="true"
	 * @generated
	 */
	typename getTypename();

	/**
	 * Sets the value of the '{@link promela.one_decl#getTypename <em>Typename</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Typename</em>' reference.
	 * @see #getTypename()
	 * @generated
	 */
	void setTypename(typename value);

	/**
	 * Returns the value of the '<em><b>Ivars</b></em>' containment reference list.
	 * The list contents are of type {@link promela.ivar}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Ivars</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Ivars</em>' containment reference list.
	 * @see promela.PromelaPackage#getone_decl_Ivars()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<ivar> getIvars();

} // one_decl
