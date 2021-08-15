/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Constant Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ConstantExpression#getConstant <em>Constant</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getConstantExpression()
 * @model
 * @generated
 */
public interface ConstantExpression extends Expression {
	/**
	 * Returns the value of the '<em><b>Constant</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Constant</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Constant</em>' containment reference.
	 * @see #setConstant(Constant)
	 * @see poosl.PooslPackage#getConstantExpression_Constant()
	 * @model containment="true" required="true"
	 * @generated
	 */
	Constant getConstant();

	/**
	 * Sets the value of the '{@link poosl.ConstantExpression#getConstant <em>Constant</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Constant</em>' containment reference.
	 * @see #getConstant()
	 * @generated
	 */
	void setConstant(Constant value);

} // ConstantExpression
