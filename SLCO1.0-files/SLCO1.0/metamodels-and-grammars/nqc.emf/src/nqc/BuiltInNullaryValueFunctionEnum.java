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
 * A representation of the literals of the enumeration '<em><b>Built In Nullary Value Function Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getBuiltInNullaryValueFunctionEnum()
 * @model
 * @generated
 */
public enum BuiltInNullaryValueFunctionEnum implements Enumerator {
	/**
	 * The '<em><b>Current Events</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CURRENT_EVENTS_VALUE
	 * @generated
	 * @ordered
	 */
	CURRENT_EVENTS(0, "CurrentEvents", "CurrentEvents"),

	/**
	 * The '<em><b>Clear All Events</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CLEAR_ALL_EVENTS_VALUE
	 * @generated
	 * @ordered
	 */
	CLEAR_ALL_EVENTS(1, "ClearAllEvents", "ClearAllEvents"),

	/**
	 * The '<em><b>Battery Level</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #BATTERY_LEVEL_VALUE
	 * @generated
	 * @ordered
	 */
	BATTERY_LEVEL(2, "BatteryLevel", "BatteryLevel"),

	/**
	 * The '<em><b>Firmware Version</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #FIRMWARE_VERSION_VALUE
	 * @generated
	 * @ordered
	 */
	FIRMWARE_VERSION(3, "FirmwareVersion", "FirmwareVersion"),

	/**
	 * The '<em><b>Program</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #PROGRAM_VALUE
	 * @generated
	 * @ordered
	 */
	PROGRAM(4, "Program", "Program"),

	/**
	 * The '<em><b>Watch</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #WATCH_VALUE
	 * @generated
	 * @ordered
	 */
	WATCH(5, "Watch", "Watch"),

	/**
	 * The '<em><b>Message</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #MESSAGE_VALUE
	 * @generated
	 * @ordered
	 */
	MESSAGE(6, "Message", "Message");

	/**
	 * The '<em><b>Current Events</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Current Events</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CURRENT_EVENTS
	 * @model name="CurrentEvents"
	 * @generated
	 * @ordered
	 */
	public static final int CURRENT_EVENTS_VALUE = 0;

	/**
	 * The '<em><b>Clear All Events</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Clear All Events</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CLEAR_ALL_EVENTS
	 * @model name="ClearAllEvents"
	 * @generated
	 * @ordered
	 */
	public static final int CLEAR_ALL_EVENTS_VALUE = 1;

	/**
	 * The '<em><b>Battery Level</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Battery Level</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #BATTERY_LEVEL
	 * @model name="BatteryLevel"
	 * @generated
	 * @ordered
	 */
	public static final int BATTERY_LEVEL_VALUE = 2;

	/**
	 * The '<em><b>Firmware Version</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Firmware Version</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #FIRMWARE_VERSION
	 * @model name="FirmwareVersion"
	 * @generated
	 * @ordered
	 */
	public static final int FIRMWARE_VERSION_VALUE = 3;

	/**
	 * The '<em><b>Program</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Program</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #PROGRAM
	 * @model name="Program"
	 * @generated
	 * @ordered
	 */
	public static final int PROGRAM_VALUE = 4;

	/**
	 * The '<em><b>Watch</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Watch</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #WATCH
	 * @model name="Watch"
	 * @generated
	 * @ordered
	 */
	public static final int WATCH_VALUE = 5;

	/**
	 * The '<em><b>Message</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Message</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #MESSAGE
	 * @model name="Message"
	 * @generated
	 * @ordered
	 */
	public static final int MESSAGE_VALUE = 6;

	/**
	 * An array of all the '<em><b>Built In Nullary Value Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final BuiltInNullaryValueFunctionEnum[] VALUES_ARRAY =
		new BuiltInNullaryValueFunctionEnum[] {
			CURRENT_EVENTS,
			CLEAR_ALL_EVENTS,
			BATTERY_LEVEL,
			FIRMWARE_VERSION,
			PROGRAM,
			WATCH,
			MESSAGE,
		};

	/**
	 * A public read-only list of all the '<em><b>Built In Nullary Value Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<BuiltInNullaryValueFunctionEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Built In Nullary Value Function Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInNullaryValueFunctionEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInNullaryValueFunctionEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Nullary Value Function Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInNullaryValueFunctionEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInNullaryValueFunctionEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Nullary Value Function Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInNullaryValueFunctionEnum get(int value) {
		switch (value) {
			case CURRENT_EVENTS_VALUE: return CURRENT_EVENTS;
			case CLEAR_ALL_EVENTS_VALUE: return CLEAR_ALL_EVENTS;
			case BATTERY_LEVEL_VALUE: return BATTERY_LEVEL;
			case FIRMWARE_VERSION_VALUE: return FIRMWARE_VERSION;
			case PROGRAM_VALUE: return PROGRAM;
			case WATCH_VALUE: return WATCH;
			case MESSAGE_VALUE: return MESSAGE;
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
	private BuiltInNullaryValueFunctionEnum(int value, String name, String literal) {
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
	
} //BuiltInNullaryValueFunctionEnum
