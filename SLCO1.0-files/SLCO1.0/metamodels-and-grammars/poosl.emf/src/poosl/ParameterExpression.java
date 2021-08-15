/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Parameter Expression</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.ParameterExpression#getParameter <em>Parameter</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getParameterExpression()
 * @model
 * @generated
 */
public interface ParameterExpression extends Expression {
	/**
	 * Returns the value of the '<em><b>Parameter</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter</em>' reference.
	 * @see #setParameter(Parameter)
	 * @see poosl.PooslPackage#getParameterExpression_Parameter()
	 * @model required="true"
	 * @generated
	 */
	Parameter getParameter();

	/**
	 * Sets the value of the '{@link poosl.ParameterExpression#getParameter <em>Parameter</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Parameter</em>' reference.
	 * @see #getParameter()
	 * @generated
	 */
	void setParameter(Parameter value);

} // ParameterExpression
