/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>ivar</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.ivar#getName <em>Name</em>}</li>
 *   <li>{@link promela.ivar#getConstant <em>Constant</em>}</li>
 *   <li>{@link promela.ivar#getAny_expr_or_ch_init <em>Any expr or ch init</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getivar()
 * @model
 * @generated
 */
public interface ivar extends EObject {
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
	 * @see promela.PromelaPackage#getivar_Name()
	 * @model required="true"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link promela.ivar#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Constant</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Constant</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Constant</em>' attribute.
	 * @see #setConstant(int)
	 * @see promela.PromelaPackage#getivar_Constant()
	 * @model
	 * @generated
	 */
	int getConstant();

	/**
	 * Sets the value of the '{@link promela.ivar#getConstant <em>Constant</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Constant</em>' attribute.
	 * @see #getConstant()
	 * @generated
	 */
	void setConstant(int value);

	/**
	 * Returns the value of the '<em><b>Any expr or ch init</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Any expr or ch init</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Any expr or ch init</em>' containment reference.
	 * @see #setAny_expr_or_ch_init(any_expr_or_ch_init)
	 * @see promela.PromelaPackage#getivar_Any_expr_or_ch_init()
	 * @model containment="true"
	 * @generated
	 */
	any_expr_or_ch_init getAny_expr_or_ch_init();

	/**
	 * Sets the value of the '{@link promela.ivar#getAny_expr_or_ch_init <em>Any expr or ch init</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Any expr or ch init</em>' containment reference.
	 * @see #getAny_expr_or_ch_init()
	 * @generated
	 */
	void setAny_expr_or_ch_init(any_expr_or_ch_init value);

} // ivar
