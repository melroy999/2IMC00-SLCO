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
 * A representation of the model object '<em><b>Primitive Data Method Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.PrimitiveDataMethodCall#getObjectExpressions <em>Object Expressions</em>}</li>
 *   <li>{@link poosl.PrimitiveDataMethodCall#getParameterExpressionLists <em>Parameter Expression Lists</em>}</li>
 *   <li>{@link poosl.PrimitiveDataMethodCall#getPrimitiveDataMethod <em>Primitive Data Method</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getPrimitiveDataMethodCall()
 * @model
 * @generated
 */
public interface PrimitiveDataMethodCall extends Expression {
	/**
	 * Returns the value of the '<em><b>Object Expressions</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Expression}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Object Expressions</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Object Expressions</em>' containment reference list.
	 * @see poosl.PooslPackage#getPrimitiveDataMethodCall_ObjectExpressions()
	 * @model containment="true" required="true"
	 * @generated
	 */
	EList<Expression> getObjectExpressions();

	/**
	 * Returns the value of the '<em><b>Parameter Expression Lists</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.ExpressionList}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter Expression Lists</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter Expression Lists</em>' containment reference list.
	 * @see poosl.PooslPackage#getPrimitiveDataMethodCall_ParameterExpressionLists()
	 * @model containment="true"
	 * @generated
	 */
	EList<ExpressionList> getParameterExpressionLists();

	/**
	 * Returns the value of the '<em><b>Primitive Data Method</b></em>' attribute.
	 * The literals are from the enumeration {@link poosl.PrimitiveDataMethodEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Primitive Data Method</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Primitive Data Method</em>' attribute.
	 * @see poosl.PrimitiveDataMethodEnum
	 * @see #setPrimitiveDataMethod(PrimitiveDataMethodEnum)
	 * @see poosl.PooslPackage#getPrimitiveDataMethodCall_PrimitiveDataMethod()
	 * @model required="true"
	 * @generated
	 */
	PrimitiveDataMethodEnum getPrimitiveDataMethod();

	/**
	 * Sets the value of the '{@link poosl.PrimitiveDataMethodCall#getPrimitiveDataMethod <em>Primitive Data Method</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Primitive Data Method</em>' attribute.
	 * @see poosl.PrimitiveDataMethodEnum
	 * @see #getPrimitiveDataMethod()
	 * @generated
	 */
	void setPrimitiveDataMethod(PrimitiveDataMethodEnum value);

} // PrimitiveDataMethodCall
