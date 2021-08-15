/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package delays;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Transition</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link delays.Transition#getDelay <em>Delay</em>}</li>
 *   <li>{@link delays.Transition#getTransition <em>Transition</em>}</li>
 * </ul>
 * </p>
 *
 * @see delays.DelaysPackage#getTransition()
 * @model
 * @generated
 */
public interface Transition extends EObject {
	/**
	 * Returns the value of the '<em><b>Delay</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Delay</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Delay</em>' attribute.
	 * @see #setDelay(int)
	 * @see delays.DelaysPackage#getTransition_Delay()
	 * @model required="true"
	 * @generated
	 */
	int getDelay();

	/**
	 * Sets the value of the '{@link delays.Transition#getDelay <em>Delay</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Delay</em>' attribute.
	 * @see #getDelay()
	 * @generated
	 */
	void setDelay(int value);

	/**
	 * Returns the value of the '<em><b>Transition</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Transition</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Transition</em>' reference.
	 * @see #setTransition(slco.Transition)
	 * @see delays.DelaysPackage#getTransition_Transition()
	 * @model required="true"
	 * @generated
	 */
	slco.Transition getTransition();

	/**
	 * Sets the value of the '{@link delays.Transition#getTransition <em>Transition</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Transition</em>' reference.
	 * @see #getTransition()
	 * @generated
	 */
	void setTransition(slco.Transition value);

} // Transition
