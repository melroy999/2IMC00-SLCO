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
 * A representation of the model object '<em><b>Data Class</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link poosl.DataClass#getVariables <em>Variables</em>}</li>
 *   <li>{@link poosl.DataClass#getMethods <em>Methods</em>}</li>
 *   <li>{@link poosl.DataClass#getSuperClass <em>Super Class</em>}</li>
 * </ul>
 * </p>
 *
 * @see poosl.PooslPackage#getDataClass()
 * @model
 * @generated
 */
public interface DataClass extends poosl.Class {
	/**
	 * Returns the value of the '<em><b>Variables</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.Variable}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Variables</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Variables</em>' containment reference list.
	 * @see poosl.PooslPackage#getDataClass_Variables()
	 * @model containment="true"
	 * @generated
	 */
	EList<Variable> getVariables();

	/**
	 * Returns the value of the '<em><b>Methods</b></em>' containment reference list.
	 * The list contents are of type {@link poosl.DataMethod}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Methods</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Methods</em>' containment reference list.
	 * @see poosl.PooslPackage#getDataClass_Methods()
	 * @model containment="true"
	 * @generated
	 */
	EList<DataMethod> getMethods();

	/**
	 * Returns the value of the '<em><b>Super Class</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Super Class</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Super Class</em>' reference.
	 * @see #setSuperClass(DataClass)
	 * @see poosl.PooslPackage#getDataClass_SuperClass()
	 * @model
	 * @generated
	 */
	DataClass getSuperClass();

	/**
	 * Sets the value of the '{@link poosl.DataClass#getSuperClass <em>Super Class</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Super Class</em>' reference.
	 * @see #getSuperClass()
	 * @generated
	 */
	void setSuperClass(DataClass value);

} // DataClass
