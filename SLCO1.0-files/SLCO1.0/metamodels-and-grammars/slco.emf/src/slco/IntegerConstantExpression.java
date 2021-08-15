/**
 */
package slco;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Integer Constant Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link slco.IntegerConstantExpression#getValue <em>Value</em>}</li>
 * </ul>
 * </p>
 *
 * @see slco.SlcoPackage#getIntegerConstantExpression()
 * @model
 * @generated
 */
public interface IntegerConstantExpression extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Value</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Value</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Value</em>' attribute.
	 * @see #setValue(int)
	 * @see slco.SlcoPackage#getIntegerConstantExpression_Value()
	 * @model unique="false" required="true" ordered="false"
	 * @generated
	 */
	int getValue();

	/**
	 * Sets the value of the '{@link slco.IntegerConstantExpression#getValue <em>Value</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Value</em>' attribute.
	 * @see #getValue()
	 * @generated
	 */
	void setValue(int value);

} // IntegerConstantExpression
