/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Array Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.ArrayExpression#getIndex <em>Index</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getArrayExpression()
 * @model
 * @generated
 */
public interface ArrayExpression extends VariableExpression {
	/**
	 * Returns the value of the '<em><b>Index</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Index</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Index</em>' reference.
	 * @see #setIndex(ValueExpression)
	 * @see nqc.NqcPackage#getArrayExpression_Index()
	 * @model
	 * @generated
	 */
	ValueExpression getIndex();

	/**
	 * Sets the value of the '{@link nqc.ArrayExpression#getIndex <em>Index</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Index</em>' reference.
	 * @see #getIndex()
	 * @generated
	 */
	void setIndex(ValueExpression value);

} // ArrayExpression
