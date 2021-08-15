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
 * A representation of the model object '<em><b>chanassert</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.chanassert#getChannel_assertion <em>Channel assertion</em>}</li>
 *   <li>{@link promela.chanassert#getVarrefs <em>Varrefs</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getchanassert()
 * @model
 * @generated
 */
public interface chanassert extends step {
	/**
	 * Returns the value of the '<em><b>Channel assertion</b></em>' attribute.
	 * The literals are from the enumeration {@link promela.channelassertion}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Channel assertion</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Channel assertion</em>' attribute.
	 * @see promela.channelassertion
	 * @see #setChannel_assertion(channelassertion)
	 * @see promela.PromelaPackage#getchanassert_Channel_assertion()
	 * @model required="true"
	 * @generated
	 */
	channelassertion getChannel_assertion();

	/**
	 * Sets the value of the '{@link promela.chanassert#getChannel_assertion <em>Channel assertion</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Channel assertion</em>' attribute.
	 * @see promela.channelassertion
	 * @see #getChannel_assertion()
	 * @generated
	 */
	void setChannel_assertion(channelassertion value);

	/**
	 * Returns the value of the '<em><b>Varrefs</b></em>' containment reference list.
	 * The list contents are of type {@link promela.varref}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Varrefs</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Varrefs</em>' containment reference list.
	 * @see promela.PromelaPackage#getchanassert_Varrefs()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<varref> getVarrefs();

} // chanassert
