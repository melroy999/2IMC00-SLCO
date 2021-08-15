/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Event Type Constant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * <ul>
 *   <li>{@link nqc.EventTypeConstant#getEventType <em>Event Type</em>}</li>
 * </ul>
 * </p>
 *
 * @see nqc.NqcPackage#getEventTypeConstant()
 * @model
 * @generated
 */
public interface EventTypeConstant extends ConstantExpression {
	/**
	 * Returns the value of the '<em><b>Event Type</b></em>' attribute.
	 * The literals are from the enumeration {@link nqc.EventTypeEnum}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Event Type</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Event Type</em>' attribute.
	 * @see nqc.EventTypeEnum
	 * @see #setEventType(EventTypeEnum)
	 * @see nqc.NqcPackage#getEventTypeConstant_EventType()
	 * @model required="true"
	 * @generated
	 */
	EventTypeEnum getEventType();

	/**
	 * Sets the value of the '{@link nqc.EventTypeConstant#getEventType <em>Event Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Event Type</em>' attribute.
	 * @see nqc.EventTypeEnum
	 * @see #getEventType()
	 * @generated
	 */
	void setEventType(EventTypeEnum value);

} // EventTypeConstant
