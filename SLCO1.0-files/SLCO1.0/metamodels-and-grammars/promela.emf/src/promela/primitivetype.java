/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>primitivetype</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.primitivetype#getPrimitivetype <em>Primitivetype</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getprimitivetype()
 * @model
 * @generated
 */
public interface primitivetype extends typename {
	/**
	 * Returns the value of the '<em><b>Primitivetype</b></em>' attribute.
	 * The literals are from the enumeration {@link promela.primitivetypeenum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Primitivetype</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Primitivetype</em>' attribute.
	 * @see promela.primitivetypeenum
	 * @see #setPrimitivetype(primitivetypeenum)
	 * @see promela.PromelaPackage#getprimitivetype_Primitivetype()
	 * @model required="true"
	 * @generated
	 */
	primitivetypeenum getPrimitivetype();

	/**
	 * Sets the value of the '{@link promela.primitivetype#getPrimitivetype <em>Primitivetype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Primitivetype</em>' attribute.
	 * @see promela.primitivetypeenum
	 * @see #getPrimitivetype()
	 * @generated
	 */
	void setPrimitivetype(primitivetypeenum value);

} // primitivetype
