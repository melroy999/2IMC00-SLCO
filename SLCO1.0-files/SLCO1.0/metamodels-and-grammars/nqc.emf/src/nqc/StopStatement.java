/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Stop Statement</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.StopStatement#getTask <em>Task</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getStopStatement()
 * @model
 * @generated
 */
public interface StopStatement extends Statement {
	/**
	 * Returns the value of the '<em><b>Task</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Task</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Task</em>' reference.
	 * @see #setTask(Task)
	 * @see nqc.NqcPackage#getStopStatement_Task()
	 * @model required="true" ordered="false"
	 * @generated
	 */
	Task getTask();

	/**
	 * Sets the value of the '{@link nqc.StopStatement#getTask <em>Task</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Task</em>' reference.
	 * @see #getTask()
	 * @generated
	 */
	void setTask(Task value);

} // StopStatement
