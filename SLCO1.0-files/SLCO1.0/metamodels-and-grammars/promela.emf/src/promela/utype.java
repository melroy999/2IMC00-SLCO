/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>utype</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.utype#getName <em>Name</em>}</li>
 *   <li>{@link promela.utype#getOne_decls <em>One decls</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getutype()
 * @model
 * @generated
 */
public interface utype extends EObject {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see promela.PromelaPackage#getutype_Name()
	 * @model required="true"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link promela.utype#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>One decls</b></em>' containment reference list.
	 * The list contents are of type {@link promela.one_decl}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>One decls</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>One decls</em>' containment reference list.
	 * @see promela.PromelaPackage#getutype_One_decls()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<one_decl> getOne_decls();

} // utype
