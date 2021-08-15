/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>spec</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link promela.spec#getName <em>Name</em>}</li>
 *   <li>{@link promela.spec#getUtypes <em>Utypes</em>}</li>
 *   <li>{@link promela.spec#getMtypes <em>Mtypes</em>}</li>
 *   <li>{@link promela.spec#getOne_decls <em>One decls</em>}</li>
 *   <li>{@link promela.spec#getProctypes <em>Proctypes</em>}</li>
 *   <li>{@link promela.spec#getInits <em>Inits</em>}</li>
 *   <li>{@link promela.spec#getNevers <em>Nevers</em>}</li>
 *   <li>{@link promela.spec#getTraces <em>Traces</em>}</li>
 *   <li>{@link promela.spec#getPrimitivetypes <em>Primitivetypes</em>}</li>
 * </ul>
 * </p>
 *
 * @see promela.PromelaPackage#getspec()
 * @model
 * @generated
 */
public interface spec extends EObject {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see promela.PromelaPackage#getspec_Name()
	 * @model required="true"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link promela.spec#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Utypes</b></em>' containment reference list.
	 * The list contents are of type {@link promela.utype}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Utypes</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Utypes</em>' containment reference list.
	 * @see promela.PromelaPackage#getspec_Utypes()
	 * @model containment="true"
	 * @generated
	 */
	EList<utype> getUtypes();

	/**
	 * Returns the value of the '<em><b>Mtypes</b></em>' containment reference list.
	 * The list contents are of type {@link promela.mtype}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Mtypes</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Mtypes</em>' containment reference list.
	 * @see promela.PromelaPackage#getspec_Mtypes()
	 * @model containment="true"
	 * @generated
	 */
	EList<mtype> getMtypes();

	/**
	 * Returns the value of the '<em><b>One decls</b></em>' containment reference list.
	 * The list contents are of type {@link promela.one_decl}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>One decls</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>One decls</em>' containment reference list.
	 * @see promela.PromelaPackage#getspec_One_decls()
	 * @model containment="true"
	 * @generated
	 */
	EList<one_decl> getOne_decls();

	/**
	 * Returns the value of the '<em><b>Proctypes</b></em>' containment reference list.
	 * The list contents are of type {@link promela.proctype}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Proctypes</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Proctypes</em>' containment reference list.
	 * @see promela.PromelaPackage#getspec_Proctypes()
	 * @model containment="true"
	 * @generated
	 */
	EList<proctype> getProctypes();

	/**
	 * Returns the value of the '<em><b>Inits</b></em>' containment reference list.
	 * The list contents are of type {@link promela.init}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Inits</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Inits</em>' containment reference list.
	 * @see promela.PromelaPackage#getspec_Inits()
	 * @model containment="true"
	 * @generated
	 */
	EList<init> getInits();

	/**
	 * Returns the value of the '<em><b>Nevers</b></em>' containment reference list.
	 * The list contents are of type {@link promela.never}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Nevers</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Nevers</em>' containment reference list.
	 * @see promela.PromelaPackage#getspec_Nevers()
	 * @model containment="true"
	 * @generated
	 */
	EList<never> getNevers();

	/**
	 * Returns the value of the '<em><b>Traces</b></em>' containment reference list.
	 * The list contents are of type {@link promela.trace}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Traces</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Traces</em>' containment reference list.
	 * @see promela.PromelaPackage#getspec_Traces()
	 * @model containment="true"
	 * @generated
	 */
	EList<trace> getTraces();

	/**
	 * Returns the value of the '<em><b>Primitivetypes</b></em>' containment reference list.
	 * The list contents are of type {@link promela.primitivetype}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Primitivetypes</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Primitivetypes</em>' containment reference list.
	 * @see promela.PromelaPackage#getspec_Primitivetypes()
	 * @model containment="true"
	 * @generated
	 */
	EList<primitivetype> getPrimitivetypes();

} // spec
