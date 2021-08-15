/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package delays;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Transition List</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link delays.TransitionList#getTransitions <em>Transitions</em>}</li>
 * </ul>
 * </p>
 *
 * @see delays.DelaysPackage#getTransitionList()
 * @model
 * @generated
 */
public interface TransitionList extends EObject {
	/**
	 * Returns the value of the '<em><b>Transitions</b></em>' containment reference list.
	 * The list contents are of type {@link delays.Transition}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Transitions</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Transitions</em>' containment reference list.
	 * @see delays.DelaysPackage#getTransitionList_Transitions()
	 * @model containment="true"
	 * @generated
	 */
	EList<Transition> getTransitions();

} // TransitionList
