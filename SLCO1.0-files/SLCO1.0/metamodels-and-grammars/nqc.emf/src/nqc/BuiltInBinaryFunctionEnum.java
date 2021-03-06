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
 * A representation of the literals of the enumeration '<em><b>Built In Binary Function Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getBuiltInBinaryFunctionEnum()
 * @model
 * @generated
 */
public enum BuiltInBinaryFunctionEnum implements Enumerator {
	/**
	 * The '<em><b>Set Sensor</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_SENSOR_VALUE
	 * @generated
	 * @ordered
	 */
	SET_SENSOR(0, "SetSensor", "SetSensor"),

	/**
	 * The '<em><b>Set Sensor Type</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_SENSOR_TYPE_VALUE
	 * @generated
	 * @ordered
	 */
	SET_SENSOR_TYPE(1, "SetSensorType", "SetSensorType"),

	/**
	 * The '<em><b>Set Sensor Mode</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_SENSOR_MODE_VALUE
	 * @generated
	 * @ordered
	 */
	SET_SENSOR_MODE(2, "SetSensorMode", "SetSensorMode"),

	/**
	 * The '<em><b>Set Output</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_OUTPUT_VALUE
	 * @generated
	 * @ordered
	 */
	SET_OUTPUT(3, "SetOutput", "SetOutput"),

	/**
	 * The '<em><b>Set Direction</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_DIRECTION_VALUE
	 * @generated
	 * @ordered
	 */
	SET_DIRECTION(4, "SetDirection", "SetDirection"),

	/**
	 * The '<em><b>Set Power</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_POWER_VALUE
	 * @generated
	 * @ordered
	 */
	SET_POWER(5, "SetPower", "SetPower"),

	/**
	 * The '<em><b>On For</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ON_FOR_VALUE
	 * @generated
	 * @ordered
	 */
	ON_FOR(6, "OnFor", "OnFor"),

	/**
	 * The '<em><b>Set Global Output</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_GLOBAL_OUTPUT_VALUE
	 * @generated
	 * @ordered
	 */
	SET_GLOBAL_OUTPUT(7, "SetGlobalOutput", "SetGlobalOutput"),

	/**
	 * The '<em><b>Set Global Direction</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_GLOBAL_DIRECTION_VALUE
	 * @generated
	 * @ordered
	 */
	SET_GLOBAL_DIRECTION(8, "SetGlobalDirection", "SetGlobalDirection"),

	/**
	 * The '<em><b>Set Max Power</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_MAX_POWER_VALUE
	 * @generated
	 * @ordered
	 */
	SET_MAX_POWER(9, "SetMaxPower", "SetMaxPower"),

	/**
	 * The '<em><b>Play Tone</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #PLAY_TONE_VALUE
	 * @generated
	 * @ordered
	 */
	PLAY_TONE(10, "PlayTone", "PlayTone"),

	/**
	 * The '<em><b>Set User Display</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_USER_DISPLAY_VALUE
	 * @generated
	 * @ordered
	 */
	SET_USER_DISPLAY(11, "SetUserDisplay", "SetUserDisplay"),

	/**
	 * The '<em><b>Set Serial Data</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_SERIAL_DATA_VALUE
	 * @generated
	 * @ordered
	 */
	SET_SERIAL_DATA(12, "SetSerialData", "SetSerialData"),

	/**
	 * The '<em><b>Send Serial</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_SERIAL_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_SERIAL(13, "SendSerial", "SendSerial"),

	/**
	 * The '<em><b>Set Spybot Ctrl Message</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_SPYBOT_CTRL_MESSAGE_VALUE
	 * @generated
	 * @ordered
	 */
	SET_SPYBOT_CTRL_MESSAGE(14, "SetSpybotCtrlMessage", "SetSpybotCtrlMessage"),

	/**
	 * The '<em><b>Send Spybot Ctrl Message</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_CTRL_MESSAGE_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_SPYBOT_CTRL_MESSAGE(15, "SendSpybotCtrlMessage", "SendSpybotCtrlMessage"),

	/**
	 * The '<em><b>Set Timer</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_TIMER_VALUE
	 * @generated
	 * @ordered
	 */
	SET_TIMER(16, "SetTimer", "SetTimer"),

	/**
	 * The '<em><b>Set Upper Limit</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_UPPER_LIMIT_VALUE
	 * @generated
	 * @ordered
	 */
	SET_UPPER_LIMIT(17, "SetUpperLimit", "SetUpperLimit"),

	/**
	 * The '<em><b>Set Lower Limit</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_LOWER_LIMIT_VALUE
	 * @generated
	 * @ordered
	 */
	SET_LOWER_LIMIT(18, "SetLowerLimit", "SetLowerLimit"),

	/**
	 * The '<em><b>Set Hysteresis</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_HYSTERESIS_VALUE
	 * @generated
	 * @ordered
	 */
	SET_HYSTERESIS(19, "SetHysteresis", "SetHysteresis"),

	/**
	 * The '<em><b>Set Click Time</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_CLICK_TIME_VALUE
	 * @generated
	 * @ordered
	 */
	SET_CLICK_TIME(20, "SetClickTime", "SetClickTime"),

	/**
	 * The '<em><b>Set Click Counter</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_CLICK_COUNTER_VALUE
	 * @generated
	 * @ordered
	 */
	SET_CLICK_COUNTER(21, "SetClickCounter", "SetClickCounter"),

	/**
	 * The '<em><b>Upload Data Log</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #UPLOAD_DATA_LOG_VALUE
	 * @generated
	 * @ordered
	 */
	UPLOAD_DATA_LOG(22, "UploadDataLog", "UploadDataLog"),

	/**
	 * The '<em><b>Set Watch</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_WATCH_VALUE
	 * @generated
	 * @ordered
	 */
	SET_WATCH(23, "SetWatch", "SetWatch");

	/**
	 * The '<em><b>Set Sensor</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Sensor</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_SENSOR
	 * @model name="SetSensor"
	 * @generated
	 * @ordered
	 */
	public static final int SET_SENSOR_VALUE = 0;

	/**
	 * The '<em><b>Set Sensor Type</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Sensor Type</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_SENSOR_TYPE
	 * @model name="SetSensorType"
	 * @generated
	 * @ordered
	 */
	public static final int SET_SENSOR_TYPE_VALUE = 1;

	/**
	 * The '<em><b>Set Sensor Mode</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Sensor Mode</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_SENSOR_MODE
	 * @model name="SetSensorMode"
	 * @generated
	 * @ordered
	 */
	public static final int SET_SENSOR_MODE_VALUE = 2;

	/**
	 * The '<em><b>Set Output</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Output</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_OUTPUT
	 * @model name="SetOutput"
	 * @generated
	 * @ordered
	 */
	public static final int SET_OUTPUT_VALUE = 3;

	/**
	 * The '<em><b>Set Direction</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Direction</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_DIRECTION
	 * @model name="SetDirection"
	 * @generated
	 * @ordered
	 */
	public static final int SET_DIRECTION_VALUE = 4;

	/**
	 * The '<em><b>Set Power</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Power</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_POWER
	 * @model name="SetPower"
	 * @generated
	 * @ordered
	 */
	public static final int SET_POWER_VALUE = 5;

	/**
	 * The '<em><b>On For</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>On For</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ON_FOR
	 * @model name="OnFor"
	 * @generated
	 * @ordered
	 */
	public static final int ON_FOR_VALUE = 6;

	/**
	 * The '<em><b>Set Global Output</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Global Output</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_GLOBAL_OUTPUT
	 * @model name="SetGlobalOutput"
	 * @generated
	 * @ordered
	 */
	public static final int SET_GLOBAL_OUTPUT_VALUE = 7;

	/**
	 * The '<em><b>Set Global Direction</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Global Direction</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_GLOBAL_DIRECTION
	 * @model name="SetGlobalDirection"
	 * @generated
	 * @ordered
	 */
	public static final int SET_GLOBAL_DIRECTION_VALUE = 8;

	/**
	 * The '<em><b>Set Max Power</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Max Power</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_MAX_POWER
	 * @model name="SetMaxPower"
	 * @generated
	 * @ordered
	 */
	public static final int SET_MAX_POWER_VALUE = 9;

	/**
	 * The '<em><b>Play Tone</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Play Tone</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #PLAY_TONE
	 * @model name="PlayTone"
	 * @generated
	 * @ordered
	 */
	public static final int PLAY_TONE_VALUE = 10;

	/**
	 * The '<em><b>Set User Display</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set User Display</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_USER_DISPLAY
	 * @model name="SetUserDisplay"
	 * @generated
	 * @ordered
	 */
	public static final int SET_USER_DISPLAY_VALUE = 11;

	/**
	 * The '<em><b>Set Serial Data</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Serial Data</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_SERIAL_DATA
	 * @model name="SetSerialData"
	 * @generated
	 * @ordered
	 */
	public static final int SET_SERIAL_DATA_VALUE = 12;

	/**
	 * The '<em><b>Send Serial</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send Serial</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_SERIAL
	 * @model name="SendSerial"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_SERIAL_VALUE = 13;

	/**
	 * The '<em><b>Set Spybot Ctrl Message</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Spybot Ctrl Message</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_SPYBOT_CTRL_MESSAGE
	 * @model name="SetSpybotCtrlMessage"
	 * @generated
	 * @ordered
	 */
	public static final int SET_SPYBOT_CTRL_MESSAGE_VALUE = 14;

	/**
	 * The '<em><b>Send Spybot Ctrl Message</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send Spybot Ctrl Message</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_CTRL_MESSAGE
	 * @model name="SendSpybotCtrlMessage"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_SPYBOT_CTRL_MESSAGE_VALUE = 15;

	/**
	 * The '<em><b>Set Timer</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Timer</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_TIMER
	 * @model name="SetTimer"
	 * @generated
	 * @ordered
	 */
	public static final int SET_TIMER_VALUE = 16;

	/**
	 * The '<em><b>Set Upper Limit</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Upper Limit</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_UPPER_LIMIT
	 * @model name="SetUpperLimit"
	 * @generated
	 * @ordered
	 */
	public static final int SET_UPPER_LIMIT_VALUE = 17;

	/**
	 * The '<em><b>Set Lower Limit</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Lower Limit</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_LOWER_LIMIT
	 * @model name="SetLowerLimit"
	 * @generated
	 * @ordered
	 */
	public static final int SET_LOWER_LIMIT_VALUE = 18;

	/**
	 * The '<em><b>Set Hysteresis</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Hysteresis</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_HYSTERESIS
	 * @model name="SetHysteresis"
	 * @generated
	 * @ordered
	 */
	public static final int SET_HYSTERESIS_VALUE = 19;

	/**
	 * The '<em><b>Set Click Time</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Click Time</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_CLICK_TIME
	 * @model name="SetClickTime"
	 * @generated
	 * @ordered
	 */
	public static final int SET_CLICK_TIME_VALUE = 20;

	/**
	 * The '<em><b>Set Click Counter</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Click Counter</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_CLICK_COUNTER
	 * @model name="SetClickCounter"
	 * @generated
	 * @ordered
	 */
	public static final int SET_CLICK_COUNTER_VALUE = 21;

	/**
	 * The '<em><b>Upload Data Log</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Upload Data Log</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #UPLOAD_DATA_LOG
	 * @model name="UploadDataLog"
	 * @generated
	 * @ordered
	 */
	public static final int UPLOAD_DATA_LOG_VALUE = 22;

	/**
	 * The '<em><b>Set Watch</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Watch</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_WATCH
	 * @model name="SetWatch"
	 * @generated
	 * @ordered
	 */
	public static final int SET_WATCH_VALUE = 23;

	/**
	 * An array of all the '<em><b>Built In Binary Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final BuiltInBinaryFunctionEnum[] VALUES_ARRAY =
		new BuiltInBinaryFunctionEnum[] {
			SET_SENSOR,
			SET_SENSOR_TYPE,
			SET_SENSOR_MODE,
			SET_OUTPUT,
			SET_DIRECTION,
			SET_POWER,
			ON_FOR,
			SET_GLOBAL_OUTPUT,
			SET_GLOBAL_DIRECTION,
			SET_MAX_POWER,
			PLAY_TONE,
			SET_USER_DISPLAY,
			SET_SERIAL_DATA,
			SEND_SERIAL,
			SET_SPYBOT_CTRL_MESSAGE,
			SEND_SPYBOT_CTRL_MESSAGE,
			SET_TIMER,
			SET_UPPER_LIMIT,
			SET_LOWER_LIMIT,
			SET_HYSTERESIS,
			SET_CLICK_TIME,
			SET_CLICK_COUNTER,
			UPLOAD_DATA_LOG,
			SET_WATCH,
		};

	/**
	 * A public read-only list of all the '<em><b>Built In Binary Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<BuiltInBinaryFunctionEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Built In Binary Function Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInBinaryFunctionEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInBinaryFunctionEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Binary Function Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInBinaryFunctionEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInBinaryFunctionEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Binary Function Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInBinaryFunctionEnum get(int value) {
		switch (value) {
			case SET_SENSOR_VALUE: return SET_SENSOR;
			case SET_SENSOR_TYPE_VALUE: return SET_SENSOR_TYPE;
			case SET_SENSOR_MODE_VALUE: return SET_SENSOR_MODE;
			case SET_OUTPUT_VALUE: return SET_OUTPUT;
			case SET_DIRECTION_VALUE: return SET_DIRECTION;
			case SET_POWER_VALUE: return SET_POWER;
			case ON_FOR_VALUE: return ON_FOR;
			case SET_GLOBAL_OUTPUT_VALUE: return SET_GLOBAL_OUTPUT;
			case SET_GLOBAL_DIRECTION_VALUE: return SET_GLOBAL_DIRECTION;
			case SET_MAX_POWER_VALUE: return SET_MAX_POWER;
			case PLAY_TONE_VALUE: return PLAY_TONE;
			case SET_USER_DISPLAY_VALUE: return SET_USER_DISPLAY;
			case SET_SERIAL_DATA_VALUE: return SET_SERIAL_DATA;
			case SEND_SERIAL_VALUE: return SEND_SERIAL;
			case SET_SPYBOT_CTRL_MESSAGE_VALUE: return SET_SPYBOT_CTRL_MESSAGE;
			case SEND_SPYBOT_CTRL_MESSAGE_VALUE: return SEND_SPYBOT_CTRL_MESSAGE;
			case SET_TIMER_VALUE: return SET_TIMER;
			case SET_UPPER_LIMIT_VALUE: return SET_UPPER_LIMIT;
			case SET_LOWER_LIMIT_VALUE: return SET_LOWER_LIMIT;
			case SET_HYSTERESIS_VALUE: return SET_HYSTERESIS;
			case SET_CLICK_TIME_VALUE: return SET_CLICK_TIME;
			case SET_CLICK_COUNTER_VALUE: return SET_CLICK_COUNTER;
			case UPLOAD_DATA_LOG_VALUE: return UPLOAD_DATA_LOG;
			case SET_WATCH_VALUE: return SET_WATCH;
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
	private BuiltInBinaryFunctionEnum(int value, String name, String literal) {
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
	
} //BuiltInBinaryFunctionEnum
