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
 * A representation of the model object '<em><b>init</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.init#getPriority <em>Priority</em>}</li>
 *   <li>{@link promela.init#getSequence <em>Sequence</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getinit()
 * @model
 * @generated
 */
public interface init extends EObject {
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
	 * @see promela.PromelaPackage#getinit_Priority()
	 * @model
	 * @generated
	 */
	int getPriority();

	/**
	 * Sets the value of the '{@link promela.init#getPriority <em>Priority</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Priority</em>' attribute.
	 * @see #getPriority()
	 * @generated
	 */
	void setPriority(int value);

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
	 * @see promela.PromelaPackage#getinit_Sequence()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<step> getSequence();

} // init
