/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Time Synchronisation</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.TimeSynchronisation#getDuration <em>Duration</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getTimeSynchronisation()
 * @model
 * @generated
 */
public interface TimeSynchronisation extends Statement {
	/**
	 * Returns the value of the '<em><b>Duration</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Duration</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Duration</em>' containment reference list.
	 * @see poosl.PooslPackage#getTimeSynchronisation_Duration()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Expression> getDuration();

} // TimeSynchronisation
