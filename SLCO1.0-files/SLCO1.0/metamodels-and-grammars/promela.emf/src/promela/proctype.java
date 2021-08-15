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
 * A representation of the model object '<em><b>proctype</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.proctype#getActive <em>Active</em>}</li>
 *   <li>{@link promela.proctype#getName <em>Name</em>}</li>
 *   <li>{@link promela.proctype#getOne_decls <em>One decls</em>}</li>
 *   <li>{@link promela.proctype#getPriority <em>Priority</em>}</li>
 *   <li>{@link promela.proctype#getEnabler <em>Enabler</em>}</li>
 *   <li>{@link promela.proctype#getSequence <em>Sequence</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getproctype()
 * @model
 * @generated
 */
public interface proctype extends EObject {
	/**
	 * Returns the value of the '<em><b>Active</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Active</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Active</em>' attribute.
	 * @see #setActive(int)
	 * @see promela.PromelaPackage#getproctype_Active()
	 * @model
	 * @generated
	 */
	int getActive();

	/**
	 * Sets the value of the '{@link promela.proctype#getActive <em>Active</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Active</em>' attribute.
	 * @see #getActive()
	 * @generated
	 */
	void setActive(int value);

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
	 * @see promela.PromelaPackage#getproctype_Name()
	 * @model required="true"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link promela.proctype#getName <em>Name</em>}' attribute.
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
	 * @see promela.PromelaPackage#getproctype_One_decls()
	 * @model containment="true"
	 * @generated
	 */
	EList<one_decl> getOne_decls();

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
	 * @see promela.PromelaPackage#getproctype_Priority()
	 * @model
	 * @generated
	 */
	int getPriority();

	/**
	 * Sets the value of the '{@link promela.proctype#getPriority <em>Priority</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Priority</em>' attribute.
	 * @see #getPriority()
	 * @generated
	 */
	void setPriority(int value);

	/**
	 * Returns the value of the '<em><b>Enabler</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Enabler</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Enabler</em>' containment reference.
	 * @see #setEnabler(expr)
	 * @see promela.PromelaPackage#getproctype_Enabler()
	 * @model containment="true"
	 * @generated
	 */
	expr getEnabler();

	/**
	 * Sets the value of the '{@link promela.proctype#getEnabler <em>Enabler</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Enabler</em>' containment reference.
	 * @see #getEnabler()
	 * @generated
	 */
	void setEnabler(expr value);

	/**
	 * Returns the value of the '<em><b>Sequence</b></em>' containment reference list.
	 * The list contents are of type {@link promela.step}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sequence</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sequence</em>' containment reference list.
	 * @see promela.PromelaPackage#getproctype_Sequence()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<step> getSequence();

} // proctype
