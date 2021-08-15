/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.eclipse.emf.common.util.Enumerator;

/**
 * <!-- begin-user-doc -->
 * A representation of the literals of the enumeration '<em><b>Event Type Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getEventTypeEnum()
 * @model
 * @generated
 */
public enum EventTypeEnum implements Enumerator {
	/**
	 * The '<em><b>EVENT TYPE PRESSED</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_PRESSED_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_PRESSED(0, "EVENT_TYPE_PRESSED", "EVENT_TYPE_PRESSED"),

	/**
	 * The '<em><b>EVENT TYPE RELEASED</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_RELEASED_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_RELEASED(1, "EVENT_TYPE_RELEASED", "EVENT_TYPE_RELEASED"),

	/**
	 * The '<em><b>EVENT TYPE PULSE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_PULSE_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_PULSE(2, "EVENT_TYPE_PULSE", "EVENT_TYPE_PULSE"),

	/**
	 * The '<em><b>EVENT TYPE EDGE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_EDGE_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_EDGE(3, "EVENT_TYPE_EDGE", "EVENT_TYPE_EDGE"),

	/**
	 * The '<em><b>EVENT TYPE FASTCHANGE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_FASTCHANGE_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_FASTCHANGE(4, "EVENT_TYPE_FASTCHANGE", "EVENT_TYPE_FASTCHANGE"),

	/**
	 * The '<em><b>EVENT TYPE LOW</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_LOW_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_LOW(5, "EVENT_TYPE_LOW", "EVENT_TYPE_LOW"),

	/**
	 * The '<em><b>EVENT TYPE NORMAL</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_NORMAL_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_NORMAL(6, "EVENT_TYPE_NORMAL", "EVENT_TYPE_NORMAL"),

	/**
	 * The '<em><b>EVENT TYPE HIGH</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_HIGH_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_HIGH(7, "EVENT_TYPE_HIGH", "EVENT_TYPE_HIGH"),

	/**
	 * The '<em><b>EVENT TYPE CLICK</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_CLICK_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_CLICK(8, "EVENT_TYPE_CLICK", "EVENT_TYPE_CLICK"),

	/**
	 * The '<em><b>EVENT TYPE DOUBLECLICK</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_DOUBLECLICK_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_DOUBLECLICK(9, "EVENT_TYPE_DOUBLECLICK", "EVENT_TYPE_DOUBLECLICK"),

	/**
	 * The '<em><b>EVENT TYPE MESSAGE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_MESSAGE_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_TYPE_MESSAGE(10, "EVENT_TYPE_MESSAGE", "EVENT_TYPE_MESSAGE");

	/**
	 * The '<em><b>EVENT TYPE PRESSED</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE PRESSED</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_PRESSED
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_PRESSED_VALUE = 0;

	/**
	 * The '<em><b>EVENT TYPE RELEASED</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE RELEASED</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_RELEASED
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_RELEASED_VALUE = 1;

	/**
	 * The '<em><b>EVENT TYPE PULSE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE PULSE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_PULSE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_PULSE_VALUE = 2;

	/**
	 * The '<em><b>EVENT TYPE EDGE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE EDGE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_EDGE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_EDGE_VALUE = 3;

	/**
	 * The '<em><b>EVENT TYPE FASTCHANGE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE FASTCHANGE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_FASTCHANGE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_FASTCHANGE_VALUE = 4;

	/**
	 * The '<em><b>EVENT TYPE LOW</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE LOW</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_LOW
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_LOW_VALUE = 5;

	/**
	 * The '<em><b>EVENT TYPE NORMAL</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE NORMAL</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_NORMAL
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_NORMAL_VALUE = 6;

	/**
	 * The '<em><b>EVENT TYPE HIGH</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE HIGH</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_HIGH
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_HIGH_VALUE = 7;

	/**
	 * The '<em><b>EVENT TYPE CLICK</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE CLICK</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_CLICK
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_CLICK_VALUE = 8;

	/**
	 * The '<em><b>EVENT TYPE DOUBLECLICK</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE DOUBLECLICK</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_DOUBLECLICK
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_DOUBLECLICK_VALUE = 9;

	/**
	 * The '<em><b>EVENT TYPE MESSAGE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EVENT TYPE MESSAGE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_TYPE_MESSAGE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_TYPE_MESSAGE_VALUE = 10;

	/**
	 * An array of all the '<em><b>Event Type Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final EventTypeEnum[] VALUES_ARRAY =
		new EventTypeEnum[] {
			EVENT_TYPE_PRESSED,
			EVENT_TYPE_RELEASED,
			EVENT_TYPE_PULSE,
			EVENT_TYPE_EDGE,
			EVENT_TYPE_FASTCHANGE,
			EVENT_TYPE_LOW,
			EVENT_TYPE_NORMAL,
			EVENT_TYPE_HIGH,
			EVENT_TYPE_CLICK,
			EVENT_TYPE_DOUBLECLICK,
			EVENT_TYPE_MESSAGE,
		};

	/**
	 * A public read-only list of all the '<em><b>Event Type Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<EventTypeEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Event Type Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static EventTypeEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			EventTypeEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Event Type Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static EventTypeEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			EventTypeEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Event Type Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static EventTypeEnum get(int value) {
		switch (value) {
			case EVENT_TYPE_PRESSED_VALUE: return EVENT_TYPE_PRESSED;
			case EVENT_TYPE_RELEASED_VALUE: return EVENT_TYPE_RELEASED;
			case EVENT_TYPE_PULSE_VALUE: return EVENT_TYPE_PULSE;
			case EVENT_TYPE_EDGE_VALUE: return EVENT_TYPE_EDGE;
			case EVENT_TYPE_FASTCHANGE_VALUE: return EVENT_TYPE_FASTCHANGE;
			case EVENT_TYPE_LOW_VALUE: return EVENT_TYPE_LOW;
			case EVENT_TYPE_NORMAL_VALUE: return EVENT_TYPE_NORMAL;
			case EVENT_TYPE_HIGH_VALUE: return EVENT_TYPE_HIGH;
			case EVENT_TYPE_CLICK_VALUE: return EVENT_TYPE_CLICK;
			case EVENT_TYPE_DOUBLECLICK_VALUE: return EVENT_TYPE_DOUBLECLICK;
			case EVENT_TYPE_MESSAGE_VALUE: return EVENT_TYPE_MESSAGE;
		}
		return null;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final int value;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final String name;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final String literal;

	/**
	 * Only this class can construct instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EventTypeEnum(int value, String name, String literal) {
		this.value = value;
		this.name = name;
		this.literal = literal;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getValue() {
	  return value;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
	  return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getLiteral() {
	  return literal;
	}

	/**
	 * Returns the literal value of the enumerator, which is its string representation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		return literal;
	}
	
} //EventTypeEnum
