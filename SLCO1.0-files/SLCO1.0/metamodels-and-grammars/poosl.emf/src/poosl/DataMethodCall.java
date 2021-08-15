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
 * A representation of the model object '<em><b>Data Method Call</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.DataMethodCall#isSuper <em>Super</em>}</li>
 *   <li>{@link poosl.DataMethodCall#getObjectExpressions <em>Object Expressions</em>}</li>
 *   <li>{@link poosl.DataMethodCall#getParameterExpressionLists <em>Parameter Expression Lists</em>}</li>
 *   <li>{@link poosl.DataMethodCall#getDataMethod <em>Data Method</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getDataMethodCall()
 * @model
 * @generated
 */
public interface DataMethodCall extends Expression {
	/**
	 * Returns the value of the '<em><b>Super</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Super</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Super</em>' attribute.
	 * @see #setSuper(boolean)
	 * @see poosl.PooslPackage#getDataMethodCall_Super()
	 * @model
	 * @generated
	 */
	boolean isSuper();

	/**
	 * Sets the value of the '{@link poosl.DataMethodCall#isSuper <em>Super</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Super</em>' attribute.
	 * @see #isSuper()
	 * @generated
	 */
	void setSuper(boolean value);

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
	 * @see poosl.PooslPackage#getDataMethodCall_ObjectExpressions()
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
	 * @see poosl.PooslPackage#getDataMethodCall_ParameterExpressionLists()
	 * @model containment="true"
	 * @generated
	 */
	EList<ExpressionList> getParameterExpressionLists();

	/**
	 * Returns the value of the '<em><b>Data Method</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Data Method</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Data Method</em>' reference.
	 * @see #setDataMethod(DataMethod)
	 * @see poosl.PooslPackage#getDataMethodCall_DataMethod()
	 * @model required="true"
	 * @generated
	 */
	DataMethod getDataMethod();

	/**
	 * Sets the value of the '{@link poosl.DataMethodCall#getDataMethod <em>Data Method</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Data Method</em>' reference.
	 * @see #getDataMethod()
	 * @generated
	 */
	void setDataMethod(DataMethod value);

} // DataMethodCall
