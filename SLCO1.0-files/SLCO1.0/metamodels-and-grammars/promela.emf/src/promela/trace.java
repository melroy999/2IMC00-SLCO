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
 * A representation of the model object '<em><b>trace</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.trace#getSequence <em>Sequence</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#gettrace()
 * @model
 * @generated
 */
public interface trace extends EObject {
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
	 * @see promela.PromelaPackage#gettrace_Sequence()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<step> getSequence();

} // trace
