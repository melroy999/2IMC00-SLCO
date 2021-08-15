/**
 */
package slco;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Textual Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.TextualStatement#getStatement <em>Statement</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getTextualStatement()
 * @model
 * @generated
 */
public interface TextualStatement extends Statement {
	/**
	 * Returns the value of the '<em><b>Statement</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Statement</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Statement</em>' attribute.
	 * @see #setStatement(String)
	 * @see slco.SlcoPackage#getTextualStatement_Statement()
	 * @model required="true"
	 * @generated
	 */
	String getStatement();

	/**
	 * Sets the value of the '{@link slco.TextualStatement#getStatement <em>Statement</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Statement</em>' attribute.
	 * @see #getStatement()
	 * @generated
	 */
	void setStatement(String value);

} // TextualStatement
