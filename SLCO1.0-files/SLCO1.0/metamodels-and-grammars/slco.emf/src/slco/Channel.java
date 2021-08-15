/**
 */
package slco;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Channel</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.Channel#getArgumentTypes <em>Argument Types</em>}</li>
 *   <li>{@link slco.Channel#getName <em>Name</em>}</li>
 *   <li>{@link slco.Channel#getChannelType <em>Channel Type</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getChannel()
 * @model abstract="true"
 * @generated
 */
public interface Channel extends EObject {
	/**
	 * Returns the value of the '<em><b>Argument Types</b></em>' containment reference list.
	 * The list contents are of type {@link slco.ArgumentType}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Argument Types</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Argument Types</em>' containment reference list.
	 * @see slco.SlcoPackage#getChannel_ArgumentTypes()
	 * @model containment="true"
	 * @generated
	 */
	EList<ArgumentType> getArgumentTypes();

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
	 * @see slco.SlcoPackage#getChannel_Name()
	 * @model required="true"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link slco.Channel#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Channel Type</b></em>' attribute.
	 * The literals are from the enumeration {@link slco.ChannelTypeEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Channel Type</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Channel Type</em>' attribute.
	 * @see slco.ChannelTypeEnum
	 * @see #setChannelType(ChannelTypeEnum)
	 * @see slco.SlcoPackage#getChannel_ChannelType()
	 * @model required="true"
	 * @generated
	 */
	ChannelTypeEnum getChannelType();

	/**
	 * Sets the value of the '{@link slco.Channel#getChannelType <em>Channel Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Channel Type</em>' attribute.
	 * @see slco.ChannelTypeEnum
	 * @see #getChannelType()
	 * @generated
	 */
	void setChannelType(ChannelTypeEnum value);

} // Channel
