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
 * A representation of the model object '<em><b>ch init</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.ch_init#getConstant <em>Constant</em>}</li>
 *   <li>{@link promela.ch_init#getTypenames <em>Typenames</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getch_init()
 * @model
 * @generated
 */
public interface ch_init extends any_expr_or_ch_init {
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
	 * @see promela.PromelaPackage#getch_init_Constant()
	 * @model
	 * @generated
	 */
	int getConstant();

	/**
	 * Sets the value of the '{@link promela.ch_init#getConstant <em>Constant</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Constant</em>' attribute.
	 * @see #getConstant()
	 * @generated
	 */
	void setConstant(int value);

	/**
	 * Returns the value of the '<em><b>Typenames</b></em>' reference list.
	 * The list contents are of type {@link promela.typename}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Typenames</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Typenames</em>' reference list.
	 * @see promela.PromelaPackage#getch_init_Typenames()
	 * @model required="true"
	 * @generated
	 */
	EList<typename> getTypenames();

} // ch_init
