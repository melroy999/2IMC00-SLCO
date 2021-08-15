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
 * A representation of the literals of the enumeration '<em><b>Built In Nullary Function Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getBuiltInNullaryFunctionEnum()
 * @model
 * @generated
 */
public enum BuiltInNullaryFunctionEnum implements Enumerator {
	/**
	 * The '<em><b>Mute Sound</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #MUTE_SOUND_VALUE
	 * @generated
	 * @ordered
	 */
	MUTE_SOUND(0, "MuteSound", "MuteSound"),

	/**
	 * The '<em><b>Unmute Sound</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #UNMUTE_SOUND_VALUE
	 * @generated
	 * @ordered
	 */
	UNMUTE_SOUND(1, "UnmuteSound", "UnmuteSound"),

	/**
	 * The '<em><b>Clear Sound</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CLEAR_SOUND_VALUE
	 * @generated
	 * @ordered
	 */
	CLEAR_SOUND(2, "ClearSound", "ClearSound"),

	/**
	 * The '<em><b>Clear Message</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CLEAR_MESSAGE_VALUE
	 * @generated
	 * @ordered
	 */
	CLEAR_MESSAGE(3, "ClearMessage", "ClearMessage"),

	/**
	 * The '<em><b>Send Spybot Ctrl Msg</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_CTRL_MSG_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_SPYBOT_CTRL_MSG(4, "SendSpybotCtrlMsg", "SendSpybotCtrlMsg"),

	/**
	 * The '<em><b>Send Spybot Ctrl Ping Msg</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_CTRL_PING_MSG_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_SPYBOT_CTRL_PING_MSG(5, "SendSpybotCtrlPingMsg", "SendSpybotCtrlPingMsg"),

	/**
	 * The '<em><b>Send Spybot Ping Msg</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_PING_MSG_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_SPYBOT_PING_MSG(6, "SendSpybotPingMsg", "SendSpybotPingMsg"),

	/**
	 * The '<em><b>Init RC Comm</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #INIT_RC_COMM_VALUE
	 * @generated
	 * @ordered
	 */
	INIT_RC_COMM(7, "InitRCComm", "InitRCComm"),

	/**
	 * The '<em><b>Send RC Msg</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_RC_MSG_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_RC_MSG(8, "SendRCMsg", "SendRCMsg"),

	/**
	 * The '<em><b>Stop All Tasks</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #STOP_ALL_TASKS_VALUE
	 * @generated
	 * @ordered
	 */
	STOP_ALL_TASKS(9, "StopAllTasks", "StopAllTasks"),

	/**
	 * The '<em><b>Sleep Now</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SLEEP_NOW_VALUE
	 * @generated
	 * @ordered
	 */
	SLEEP_NOW(10, "SleepNow", "SleepNow");

	/**
	 * The '<em><b>Mute Sound</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Mute Sound</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #MUTE_SOUND
	 * @model name="MuteSound"
	 * @generated
	 * @ordered
	 */
	public static final int MUTE_SOUND_VALUE = 0;

	/**
	 * The '<em><b>Unmute Sound</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Unmute Sound</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #UNMUTE_SOUND
	 * @model name="UnmuteSound"
	 * @generated
	 * @ordered
	 */
	public static final int UNMUTE_SOUND_VALUE = 1;

	/**
	 * The '<em><b>Clear Sound</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Clear Sound</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CLEAR_SOUND
	 * @model name="ClearSound"
	 * @generated
	 * @ordered
	 */
	public static final int CLEAR_SOUND_VALUE = 2;

	/**
	 * The '<em><b>Clear Message</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Clear Message</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CLEAR_MESSAGE
	 * @model name="ClearMessage"
	 * @generated
	 * @ordered
	 */
	public static final int CLEAR_MESSAGE_VALUE = 3;

	/**
	 * The '<em><b>Send Spybot Ctrl Msg</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send Spybot Ctrl Msg</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_CTRL_MSG
	 * @model name="SendSpybotCtrlMsg"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_SPYBOT_CTRL_MSG_VALUE = 4;

	/**
	 * The '<em><b>Send Spybot Ctrl Ping Msg</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send Spybot Ctrl Ping Msg</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_CTRL_PING_MSG
	 * @model name="SendSpybotCtrlPingMsg"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_SPYBOT_CTRL_PING_MSG_VALUE = 5;

	/**
	 * The '<em><b>Send Spybot Ping Msg</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send Spybot Ping Msg</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_PING_MSG
	 * @model name="SendSpybotPingMsg"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_SPYBOT_PING_MSG_VALUE = 6;

	/**
	 * The '<em><b>Init RC Comm</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Init RC Comm</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #INIT_RC_COMM
	 * @model name="InitRCComm"
	 * @generated
	 * @ordered
	 */
	public static final int INIT_RC_COMM_VALUE = 7;

	/**
	 * The '<em><b>Send RC Msg</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send RC Msg</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_RC_MSG
	 * @model name="SendRCMsg"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_RC_MSG_VALUE = 8;

	/**
	 * The '<em><b>Stop All Tasks</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Stop All Tasks</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #STOP_ALL_TASKS
	 * @model name="StopAllTasks"
	 * @generated
	 * @ordered
	 */
	public static final int STOP_ALL_TASKS_VALUE = 9;

	/**
	 * The '<em><b>Sleep Now</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Sleep Now</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SLEEP_NOW
	 * @model name="SleepNow"
	 * @generated
	 * @ordered
	 */
	public static final int SLEEP_NOW_VALUE = 10;

	/**
	 * An array of all the '<em><b>Built In Nullary Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final BuiltInNullaryFunctionEnum[] VALUES_ARRAY =
		new BuiltInNullaryFunctionEnum[] {
			MUTE_SOUND,
			UNMUTE_SOUND,
			CLEAR_SOUND,
			CLEAR_MESSAGE,
			SEND_SPYBOT_CTRL_MSG,
			SEND_SPYBOT_CTRL_PING_MSG,
			SEND_SPYBOT_PING_MSG,
			INIT_RC_COMM,
			SEND_RC_MSG,
			STOP_ALL_TASKS,
			SLEEP_NOW,
		};

	/**
	 * A public read-only list of all the '<em><b>Built In Nullary Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<BuiltInNullaryFunctionEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Built In Nullary Function Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInNullaryFunctionEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInNullaryFunctionEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Nullary Function Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInNullaryFunctionEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInNullaryFunctionEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Nullary Function Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInNullaryFunctionEnum get(int value) {
		switch (value) {
			case MUTE_SOUND_VALUE: return MUTE_SOUND;
			case UNMUTE_SOUND_VALUE: return UNMUTE_SOUND;
			case CLEAR_SOUND_VALUE: return CLEAR_SOUND;
			case CLEAR_MESSAGE_VALUE: return CLEAR_MESSAGE;
			case SEND_SPYBOT_CTRL_MSG_VALUE: return SEND_SPYBOT_CTRL_MSG;
			case SEND_SPYBOT_CTRL_PING_MSG_VALUE: return SEND_SPYBOT_CTRL_PING_MSG;
			case SEND_SPYBOT_PING_MSG_VALUE: return SEND_SPYBOT_PING_MSG;
			case INIT_RC_COMM_VALUE: return INIT_RC_COMM;
			case SEND_RC_MSG_VALUE: return SEND_RC_MSG;
			case STOP_ALL_TASKS_VALUE: return STOP_ALL_TASKS;
			case SLEEP_NOW_VALUE: return SLEEP_NOW;
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
	private BuiltInNullaryFunctionEnum(int value, String name, String literal) {
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
	
} //BuiltInNullaryFunctionEnum
