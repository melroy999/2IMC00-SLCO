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
 * A representation of the literals of the enumeration '<em><b>Built In Unary Value Function Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getBuiltInUnaryValueFunctionEnum()
 * @model
 * @generated
 */
public enum BuiltInUnaryValueFunctionEnum implements Enumerator {
	/**
	 * The '<em><b>Sensor Value</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_VALUE_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_VALUE(0, "SensorValue", "SensorValue"),

	/**
	 * The '<em><b>Sensor Type</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_TYPE(1, "SensorType", "SensorType"),

	/**
	 * The '<em><b>Sensor Mode</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_MODE(2, "SensorMode", "SensorMode"),

	/**
	 * The '<em><b>Sensor Value Bool</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_VALUE_BOOL_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_VALUE_BOOL(3, "SensorValueBool", "SensorValueBool"),

	/**
	 * The '<em><b>Sensor Value Raw</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_VALUE_RAW_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_VALUE_RAW(4, "SensorValueRaw", "SensorValueRaw"),

	/**
	 * The '<em><b>Output Status</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OUTPUT_STATUS_VALUE
	 * @generated
	 * @ordered
	 */
	OUTPUT_STATUS(5, "OutputStatus", "OutputStatus"),

	/**
	 * The '<em><b>Global Output Status</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #GLOBAL_OUTPUT_STATUS_VALUE
	 * @generated
	 * @ordered
	 */
	GLOBAL_OUTPUT_STATUS(6, "GlobalOutputStatus", "GlobalOutputStatus"),

	/**
	 * The '<em><b>Serial Data</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_DATA_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_DATA(7, "SerialData", "SerialData"),

	/**
	 * The '<em><b>Timer</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #TIMER_VALUE
	 * @generated
	 * @ordered
	 */
	TIMER(8, "Timer", "Timer"),

	/**
	 * The '<em><b>Fast Timer</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #FAST_TIMER_VALUE
	 * @generated
	 * @ordered
	 */
	FAST_TIMER(9, "FastTimer", "FastTimer"),

	/**
	 * The '<em><b>Counter</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #COUNTER_VALUE
	 * @generated
	 * @ordered
	 */
	COUNTER(10, "Counter", "Counter"),

	/**
	 * The '<em><b>Active Events</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ACTIVE_EVENTS_VALUE
	 * @generated
	 * @ordered
	 */
	ACTIVE_EVENTS(11, "ActiveEvents", "ActiveEvents"),

	/**
	 * The '<em><b>Clear Event</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CLEAR_EVENT_VALUE
	 * @generated
	 * @ordered
	 */
	CLEAR_EVENT(12, "ClearEvent", "ClearEvent"),

	/**
	 * The '<em><b>Event State</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_STATE_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT_STATE(13, "EventState", "EventState"),

	/**
	 * The '<em><b>Upper Limit</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #UPPER_LIMIT_VALUE
	 * @generated
	 * @ordered
	 */
	UPPER_LIMIT(14, "UpperLimit", "UpperLimit"),

	/**
	 * The '<em><b>Lower Limit</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #LOWER_LIMIT_VALUE
	 * @generated
	 * @ordered
	 */
	LOWER_LIMIT(15, "LowerLimit", "LowerLimit"),

	/**
	 * The '<em><b>Hysteresis</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #HYSTERESIS_VALUE
	 * @generated
	 * @ordered
	 */
	HYSTERESIS(16, "Hysteresis", "Hysteresis"),

	/**
	 * The '<em><b>Click Time</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CLICK_TIME_VALUE
	 * @generated
	 * @ordered
	 */
	CLICK_TIME(17, "ClickTime", "ClickTime"),

	/**
	 * The '<em><b>Click Counter</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CLICK_COUNTER_VALUE
	 * @generated
	 * @ordered
	 */
	CLICK_COUNTER(18, "ClickCounter", "ClickCounter"),

	/**
	 * The '<em><b>Random</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #RANDOM_VALUE
	 * @generated
	 * @ordered
	 */
	RANDOM(19, "Random", "Random"),

	/**
	 * The '<em><b>Indirect</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #INDIRECT_VALUE
	 * @generated
	 * @ordered
	 */
	INDIRECT(20, "Indirect", "Indirect");

	/**
	 * The '<em><b>Sensor Value</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Sensor Value</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_VALUE
	 * @model name="SensorValue"
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_VALUE_VALUE = 0;

	/**
	 * The '<em><b>Sensor Type</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Sensor Type</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE
	 * @model name="SensorType"
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_TYPE_VALUE = 1;

	/**
	 * The '<em><b>Sensor Mode</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Sensor Mode</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE
	 * @model name="SensorMode"
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_MODE_VALUE = 2;

	/**
	 * The '<em><b>Sensor Value Bool</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Sensor Value Bool</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_VALUE_BOOL
	 * @model name="SensorValueBool"
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_VALUE_BOOL_VALUE = 3;

	/**
	 * The '<em><b>Sensor Value Raw</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Sensor Value Raw</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_VALUE_RAW
	 * @model name="SensorValueRaw"
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_VALUE_RAW_VALUE = 4;

	/**
	 * The '<em><b>Output Status</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Output Status</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OUTPUT_STATUS
	 * @model name="OutputStatus"
	 * @generated
	 * @ordered
	 */
	public static final int OUTPUT_STATUS_VALUE = 5;

	/**
	 * The '<em><b>Global Output Status</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Global Output Status</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #GLOBAL_OUTPUT_STATUS
	 * @model name="GlobalOutputStatus"
	 * @generated
	 * @ordered
	 */
	public static final int GLOBAL_OUTPUT_STATUS_VALUE = 6;

	/**
	 * The '<em><b>Serial Data</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Serial Data</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_DATA
	 * @model name="SerialData"
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_DATA_VALUE = 7;

	/**
	 * The '<em><b>Timer</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Timer</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #TIMER
	 * @model name="Timer"
	 * @generated
	 * @ordered
	 */
	public static final int TIMER_VALUE = 8;

	/**
	 * The '<em><b>Fast Timer</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Fast Timer</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #FAST_TIMER
	 * @model name="FastTimer"
	 * @generated
	 * @ordered
	 */
	public static final int FAST_TIMER_VALUE = 9;

	/**
	 * The '<em><b>Counter</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Counter</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #COUNTER
	 * @model name="Counter"
	 * @generated
	 * @ordered
	 */
	public static final int COUNTER_VALUE = 10;

	/**
	 * The '<em><b>Active Events</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Active Events</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ACTIVE_EVENTS
	 * @model name="ActiveEvents"
	 * @generated
	 * @ordered
	 */
	public static final int ACTIVE_EVENTS_VALUE = 11;

	/**
	 * The '<em><b>Clear Event</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Clear Event</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CLEAR_EVENT
	 * @model name="ClearEvent"
	 * @generated
	 * @ordered
	 */
	public static final int CLEAR_EVENT_VALUE = 12;

	/**
	 * The '<em><b>Event State</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Event State</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT_STATE
	 * @model name="EventState"
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_STATE_VALUE = 13;

	/**
	 * The '<em><b>Upper Limit</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Upper Limit</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #UPPER_LIMIT
	 * @model name="UpperLimit"
	 * @generated
	 * @ordered
	 */
	public static final int UPPER_LIMIT_VALUE = 14;

	/**
	 * The '<em><b>Lower Limit</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Lower Limit</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #LOWER_LIMIT
	 * @model name="LowerLimit"
	 * @generated
	 * @ordered
	 */
	public static final int LOWER_LIMIT_VALUE = 15;

	/**
	 * The '<em><b>Hysteresis</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Hysteresis</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #HYSTERESIS
	 * @model name="Hysteresis"
	 * @generated
	 * @ordered
	 */
	public static final int HYSTERESIS_VALUE = 16;

	/**
	 * The '<em><b>Click Time</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Click Time</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CLICK_TIME
	 * @model name="ClickTime"
	 * @generated
	 * @ordered
	 */
	public static final int CLICK_TIME_VALUE = 17;

	/**
	 * The '<em><b>Click Counter</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Click Counter</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CLICK_COUNTER
	 * @model name="ClickCounter"
	 * @generated
	 * @ordered
	 */
	public static final int CLICK_COUNTER_VALUE = 18;

	/**
	 * The '<em><b>Random</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Random</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #RANDOM
	 * @model name="Random"
	 * @generated
	 * @ordered
	 */
	public static final int RANDOM_VALUE = 19;

	/**
	 * The '<em><b>Indirect</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Indirect</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #INDIRECT
	 * @model name="Indirect"
	 * @generated
	 * @ordered
	 */
	public static final int INDIRECT_VALUE = 20;

	/**
	 * An array of all the '<em><b>Built In Unary Value Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final BuiltInUnaryValueFunctionEnum[] VALUES_ARRAY =
		new BuiltInUnaryValueFunctionEnum[] {
			SENSOR_VALUE,
			SENSOR_TYPE,
			SENSOR_MODE,
			SENSOR_VALUE_BOOL,
			SENSOR_VALUE_RAW,
			OUTPUT_STATUS,
			GLOBAL_OUTPUT_STATUS,
			SERIAL_DATA,
			TIMER,
			FAST_TIMER,
			COUNTER,
			ACTIVE_EVENTS,
			CLEAR_EVENT,
			EVENT_STATE,
			UPPER_LIMIT,
			LOWER_LIMIT,
			HYSTERESIS,
			CLICK_TIME,
			CLICK_COUNTER,
			RANDOM,
			INDIRECT,
		};

	/**
	 * A public read-only list of all the '<em><b>Built In Unary Value Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<BuiltInUnaryValueFunctionEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Built In Unary Value Function Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInUnaryValueFunctionEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInUnaryValueFunctionEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Unary Value Function Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInUnaryValueFunctionEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInUnaryValueFunctionEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Unary Value Function Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInUnaryValueFunctionEnum get(int value) {
		switch (value) {
			case SENSOR_VALUE_VALUE: return SENSOR_VALUE;
			case SENSOR_TYPE_VALUE: return SENSOR_TYPE;
			case SENSOR_MODE_VALUE: return SENSOR_MODE;
			case SENSOR_VALUE_BOOL_VALUE: return SENSOR_VALUE_BOOL;
			case SENSOR_VALUE_RAW_VALUE: return SENSOR_VALUE_RAW;
			case OUTPUT_STATUS_VALUE: return OUTPUT_STATUS;
			case GLOBAL_OUTPUT_STATUS_VALUE: return GLOBAL_OUTPUT_STATUS;
			case SERIAL_DATA_VALUE: return SERIAL_DATA;
			case TIMER_VALUE: return TIMER;
			case FAST_TIMER_VALUE: return FAST_TIMER;
			case COUNTER_VALUE: return COUNTER;
			case ACTIVE_EVENTS_VALUE: return ACTIVE_EVENTS;
			case CLEAR_EVENT_VALUE: return CLEAR_EVENT;
			case EVENT_STATE_VALUE: return EVENT_STATE;
			case UPPER_LIMIT_VALUE: return UPPER_LIMIT;
			case LOWER_LIMIT_VALUE: return LOWER_LIMIT;
			case HYSTERESIS_VALUE: return HYSTERESIS;
			case CLICK_TIME_VALUE: return CLICK_TIME;
			case CLICK_COUNTER_VALUE: return CLICK_COUNTER;
			case RANDOM_VALUE: return RANDOM;
			case INDIRECT_VALUE: return INDIRECT;
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
	private BuiltInUnaryValueFunctionEnum(int value, String name, String literal) {
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
	
} //BuiltInUnaryValueFunctionEnum
