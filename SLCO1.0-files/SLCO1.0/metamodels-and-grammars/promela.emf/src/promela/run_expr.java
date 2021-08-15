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
 * A representation of the model object '<em><b>run expr</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.run_expr#getProc <em>Proc</em>}</li>
 *   <li>{@link promela.run_expr#getArg_lst <em>Arg lst</em>}</li>
 *   <li>{@link promela.run_expr#getPriority <em>Priority</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getrun_expr()
 * @model
 * @generated
 */
public interface run_expr extends any_expr {
	/**
	 * Returns the value of the '<em><b>Proc</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Proc</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Proc</em>' reference.
	 * @see #setProc(proctype)
	 * @see promela.PromelaPackage#getrun_expr_Proc()
	 * @model required="true"
	 * @generated
	 */
	proctype getProc();

	/**
	 * Sets the value of the '{@link promela.run_expr#getProc <em>Proc</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Proc</em>' reference.
	 * @see #getProc()
	 * @generated
	 */
	void setProc(proctype value);

	/**
	 * Returns the value of the '<em><b>Arg lst</b></em>' containment reference list.
	 * The list contents are of type {@link promela.any_expr}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Arg lst</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Arg lst</em>' containment reference list.
	 * @see promela.PromelaPackage#getrun_expr_Arg_lst()
	 * @model containment="true"
	 * @generated
	 */
	EList<any_expr> getArg_lst();

	/**
	 * Returns the value of the '<em><b>Priority</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Priority</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Priority</em>' attribute.
	 * @see #setPriority(int)
	 * @see promela.PromelaPackage#getrun_expr_Priority()
	 * @model
	 * @generated
	 */
	int getPriority();

	/**
	 * Sets the value of the '{@link promela.run_expr#getPriority <em>Priority</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Priority</em>' attribute.
	 * @see #getPriority()
	 * @generated
	 */
	void setPriority(int value);

} // run_expr
