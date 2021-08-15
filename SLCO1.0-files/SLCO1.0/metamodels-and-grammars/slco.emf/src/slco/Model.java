/**
 */
package slco;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Model</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.Model#getClasses <em>Classes</em>}</li>
 *   <li>{@link slco.Model#getObjects <em>Objects</em>}</li>
 *   <li>{@link slco.Model#getName <em>Name</em>}</li>
 *   <li>{@link slco.Model#getChannels <em>Channels</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getModel()
 * @model
 * @generated
 */
public interface Model extends EObject {
	/**
	 * Returns the value of the '<em><b>Classes</b></em>' containment reference list.
	 * The list contents are of type {@link slco.Class}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Classes</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Classes</em>' containment reference list.
	 * @see slco.SlcoPackage#getModel_Classes()
	 * @model containment="true"
	 * @generated
	 */
	EList<slco.Class> getClasses();

	/**
	 * Returns the value of the '<em><b>Objects</b></em>' containment reference list.
	 * The list contents are of type {@link slco.Object}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Objects</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Objects</em>' containment reference list.
	 * @see slco.SlcoPackage#getModel_Objects()
	 * @model containment="true"
	 * @generated
	 */
	EList<slco.Object> getObjects();

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
	 * @see slco.SlcoPackage#getModel_Name()
	 * @model id="true" required="true"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link slco.Model#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Channels</b></em>' containment reference list.
	 * The list contents are of type {@link slco.Channel}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Channels</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Channels</em>' containment reference list.
	 * @see slco.SlcoPackage#getModel_Channels()
	 * @model containment="true"
	 * @generated
	 */
	EList<Channel> getChannels();

} // Model
