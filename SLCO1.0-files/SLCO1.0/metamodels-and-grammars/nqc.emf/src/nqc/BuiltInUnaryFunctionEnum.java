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
 * A representation of the literals of the enumeration '<em><b>Built In Unary Function Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getBuiltInUnaryFunctionEnum()
 * @model
 * @generated
 */
public enum BuiltInUnaryFunctionEnum implements Enumerator {
	/**
	 * The '<em><b>Clear Sensor</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CLEAR_SENSOR_VALUE
	 * @generated
	 * @ordered
	 */
	CLEAR_SENSOR(0, "ClearSensor", "ClearSensor"),

	/**
	 * The '<em><b>On</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ON_VALUE
	 * @generated
	 * @ordered
	 */
	ON(1, "On", "On"),

	/**
	 * The '<em><b>Off</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OFF_VALUE
	 * @generated
	 * @ordered
	 */
	OFF(2, "Off", "Off"),

	/**
	 * The '<em><b>Float</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #FLOAT_VALUE
	 * @generated
	 * @ordered
	 */
	FLOAT(3, "Float", "Float"),

	/**
	 * The '<em><b>Fwd</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #FWD_VALUE
	 * @generated
	 * @ordered
	 */
	FWD(4, "Fwd", "Fwd"),

	/**
	 * The '<em><b>Rev</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #REV_VALUE
	 * @generated
	 * @ordered
	 */
	REV(5, "Rev", "Rev"),

	/**
	 * The '<em><b>Toggle</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #TOGGLE_VALUE
	 * @generated
	 * @ordered
	 */
	TOGGLE(6, "Toggle", "Toggle"),

	/**
	 * The '<em><b>On Fwd</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ON_FWD_VALUE
	 * @generated
	 * @ordered
	 */
	ON_FWD(7, "OnFwd", "OnFwd"),

	/**
	 * The '<em><b>On Rev</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ON_REV_VALUE
	 * @generated
	 * @ordered
	 */
	ON_REV(8, "OnRev", "OnRev"),

	/**
	 * The '<em><b>Enable Output</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ENABLE_OUTPUT_VALUE
	 * @generated
	 * @ordered
	 */
	ENABLE_OUTPUT(9, "EnableOutput", "EnableOutput"),

	/**
	 * The '<em><b>Disable Output</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #DISABLE_OUTPUT_VALUE
	 * @generated
	 * @ordered
	 */
	DISABLE_OUTPUT(10, "DisableOutput", "DisableOutput"),

	/**
	 * The '<em><b>Invert Output</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #INVERT_OUTPUT_VALUE
	 * @generated
	 * @ordered
	 */
	INVERT_OUTPUT(11, "InvertOutput", "InvertOutput"),

	/**
	 * The '<em><b>Obvert Output</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OBVERT_OUTPUT_VALUE
	 * @generated
	 * @ordered
	 */
	OBVERT_OUTPUT(12, "ObvertOutput", "ObvertOutput"),

	/**
	 * The '<em><b>Play Sound</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #PLAY_SOUND_VALUE
	 * @generated
	 * @ordered
	 */
	PLAY_SOUND(13, "PlaySound", "PlaySound"),

	/**
	 * The '<em><b>Select Display</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SELECT_DISPLAY_VALUE
	 * @generated
	 * @ordered
	 */
	SELECT_DISPLAY(14, "SelectDisplay", "SelectDisplay"),

	/**
	 * The '<em><b>Send Message</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_MESSAGE_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_MESSAGE(15, "SendMessage", "SendMessage"),

	/**
	 * The '<em><b>Set Tx Power</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_TX_POWER_VALUE
	 * @generated
	 * @ordered
	 */
	SET_TX_POWER(16, "SetTxPower", "SetTxPower"),

	/**
	 * The '<em><b>Set Serial Comm</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_SERIAL_COMM_VALUE
	 * @generated
	 * @ordered
	 */
	SET_SERIAL_COMM(17, "SetSerialComm", "SetSerialComm"),

	/**
	 * The '<em><b>Set Serial Packet</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_SERIAL_PACKET_VALUE
	 * @generated
	 * @ordered
	 */
	SET_SERIAL_PACKET(18, "SetSerialPacket", "SetSerialPacket"),

	/**
	 * The '<em><b>Init Spybot Comm</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #INIT_SPYBOT_COMM_VALUE
	 * @generated
	 * @ordered
	 */
	INIT_SPYBOT_COMM(19, "InitSpybotComm", "InitSpybotComm"),

	/**
	 * The '<em><b>Send Spybot Msg</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_MSG_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_SPYBOT_MSG(20, "SendSpybotMsg", "SendSpybotMsg"),

	/**
	 * The '<em><b>Set Spybot Ctrl Ping Message</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_SPYBOT_CTRL_PING_MESSAGE_VALUE
	 * @generated
	 * @ordered
	 */
	SET_SPYBOT_CTRL_PING_MESSAGE(21, "SetSpybotCtrlPingMessage", "SetSpybotCtrlPingMessage"),

	/**
	 * The '<em><b>Send Spybot Ctrl Ping Message</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_CTRL_PING_MESSAGE_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_SPYBOT_CTRL_PING_MESSAGE(22, "SendSpybotCtrlPingMessage", "SendSpybotCtrlPingMessage"),

	/**
	 * The '<em><b>Clear Timer</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CLEAR_TIMER_VALUE
	 * @generated
	 * @ordered
	 */
	CLEAR_TIMER(23, "ClearTimer", "ClearTimer"),

	/**
	 * The '<em><b>Clear Counter</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CLEAR_COUNTER_VALUE
	 * @generated
	 * @ordered
	 */
	CLEAR_COUNTER(24, "ClearCounter", "ClearCounter"),

	/**
	 * The '<em><b>Inc Counter</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #INC_COUNTER_VALUE
	 * @generated
	 * @ordered
	 */
	INC_COUNTER(25, "IncCounter", "IncCounter"),

	/**
	 * The '<em><b>Dec Counter</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #DEC_COUNTER_VALUE
	 * @generated
	 * @ordered
	 */
	DEC_COUNTER(26, "DecCounter", "DecCounter"),

	/**
	 * The '<em><b>Set Priority</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_PRIORITY_VALUE
	 * @generated
	 * @ordered
	 */
	SET_PRIORITY(27, "SetPriority", "SetPriority"),

	/**
	 * The '<em><b>Event</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EVENT_VALUE
	 * @generated
	 * @ordered
	 */
	EVENT(28, "Event", "Event"),

	/**
	 * The '<em><b>Create Datalog</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CREATE_DATALOG_VALUE
	 * @generated
	 * @ordered
	 */
	CREATE_DATALOG(29, "CreateDatalog", "CreateDatalog"),

	/**
	 * The '<em><b>Add To Datalog</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ADD_TO_DATALOG_VALUE
	 * @generated
	 * @ordered
	 */
	ADD_TO_DATALOG(30, "AddToDatalog", "AddToDatalog"),

	/**
	 * The '<em><b>Wait</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #WAIT_VALUE
	 * @generated
	 * @ordered
	 */
	WAIT(31, "Wait", "Wait"),

	/**
	 * The '<em><b>Set Random Seed</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_RANDOM_SEED_VALUE
	 * @generated
	 * @ordered
	 */
	SET_RANDOM_SEED(32, "SetRandomSeed", "SetRandomSeed"),

	/**
	 * The '<em><b>Set Sleep Time</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_SLEEP_TIME_VALUE
	 * @generated
	 * @ordered
	 */
	SET_SLEEP_TIME(33, "SetSleepTime", "SetSleepTime"),

	/**
	 * The '<em><b>Select Program</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SELECT_PROGRAM_VALUE
	 * @generated
	 * @ordered
	 */
	SELECT_PROGRAM(34, "SelectProgram", "SelectProgram");

	/**
	 * The '<em><b>Clear Sensor</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Clear Sensor</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CLEAR_SENSOR
	 * @model name="ClearSensor"
	 * @generated
	 * @ordered
	 */
	public static final int CLEAR_SENSOR_VALUE = 0;

	/**
	 * The '<em><b>On</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>On</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ON
	 * @model name="On"
	 * @generated
	 * @ordered
	 */
	public static final int ON_VALUE = 1;

	/**
	 * The '<em><b>Off</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Off</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OFF
	 * @model name="Off"
	 * @generated
	 * @ordered
	 */
	public static final int OFF_VALUE = 2;

	/**
	 * The '<em><b>Float</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Float</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #FLOAT
	 * @model name="Float"
	 * @generated
	 * @ordered
	 */
	public static final int FLOAT_VALUE = 3;

	/**
	 * The '<em><b>Fwd</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Fwd</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #FWD
	 * @model name="Fwd"
	 * @generated
	 * @ordered
	 */
	public static final int FWD_VALUE = 4;

	/**
	 * The '<em><b>Rev</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Rev</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #REV
	 * @model name="Rev"
	 * @generated
	 * @ordered
	 */
	public static final int REV_VALUE = 5;

	/**
	 * The '<em><b>Toggle</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Toggle</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #TOGGLE
	 * @model name="Toggle"
	 * @generated
	 * @ordered
	 */
	public static final int TOGGLE_VALUE = 6;

	/**
	 * The '<em><b>On Fwd</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>On Fwd</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ON_FWD
	 * @model name="OnFwd"
	 * @generated
	 * @ordered
	 */
	public static final int ON_FWD_VALUE = 7;

	/**
	 * The '<em><b>On Rev</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>On Rev</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ON_REV
	 * @model name="OnRev"
	 * @generated
	 * @ordered
	 */
	public static final int ON_REV_VALUE = 8;

	/**
	 * The '<em><b>Enable Output</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Enable Output</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ENABLE_OUTPUT
	 * @model name="EnableOutput"
	 * @generated
	 * @ordered
	 */
	public static final int ENABLE_OUTPUT_VALUE = 9;

	/**
	 * The '<em><b>Disable Output</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Disable Output</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #DISABLE_OUTPUT
	 * @model name="DisableOutput"
	 * @generated
	 * @ordered
	 */
	public static final int DISABLE_OUTPUT_VALUE = 10;

	/**
	 * The '<em><b>Invert Output</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Invert Output</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #INVERT_OUTPUT
	 * @model name="InvertOutput"
	 * @generated
	 * @ordered
	 */
	public static final int INVERT_OUTPUT_VALUE = 11;

	/**
	 * The '<em><b>Obvert Output</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Obvert Output</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OBVERT_OUTPUT
	 * @model name="ObvertOutput"
	 * @generated
	 * @ordered
	 */
	public static final int OBVERT_OUTPUT_VALUE = 12;

	/**
	 * The '<em><b>Play Sound</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Play Sound</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #PLAY_SOUND
	 * @model name="PlaySound"
	 * @generated
	 * @ordered
	 */
	public static final int PLAY_SOUND_VALUE = 13;

	/**
	 * The '<em><b>Select Display</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Select Display</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SELECT_DISPLAY
	 * @model name="SelectDisplay"
	 * @generated
	 * @ordered
	 */
	public static final int SELECT_DISPLAY_VALUE = 14;

	/**
	 * The '<em><b>Send Message</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send Message</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_MESSAGE
	 * @model name="SendMessage"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_MESSAGE_VALUE = 15;

	/**
	 * The '<em><b>Set Tx Power</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Tx Power</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_TX_POWER
	 * @model name="SetTxPower"
	 * @generated
	 * @ordered
	 */
	public static final int SET_TX_POWER_VALUE = 16;

	/**
	 * The '<em><b>Set Serial Comm</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Serial Comm</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_SERIAL_COMM
	 * @model name="SetSerialComm"
	 * @generated
	 * @ordered
	 */
	public static final int SET_SERIAL_COMM_VALUE = 17;

	/**
	 * The '<em><b>Set Serial Packet</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Serial Packet</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_SERIAL_PACKET
	 * @model name="SetSerialPacket"
	 * @generated
	 * @ordered
	 */
	public static final int SET_SERIAL_PACKET_VALUE = 18;

	/**
	 * The '<em><b>Init Spybot Comm</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Init Spybot Comm</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #INIT_SPYBOT_COMM
	 * @model name="InitSpybotComm"
	 * @generated
	 * @ordered
	 */
	public static final int INIT_SPYBOT_COMM_VALUE = 19;

	/**
	 * The '<em><b>Send Spybot Msg</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send Spybot Msg</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_MSG
	 * @model name="SendSpybotMsg"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_SPYBOT_MSG_VALUE = 20;

	/**
	 * The '<em><b>Set Spybot Ctrl Ping Message</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Spybot Ctrl Ping Message</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_SPYBOT_CTRL_PING_MESSAGE
	 * @model name="SetSpybotCtrlPingMessage"
	 * @generated
	 * @ordered
	 */
	public static final int SET_SPYBOT_CTRL_PING_MESSAGE_VALUE = 21;

	/**
	 * The '<em><b>Send Spybot Ctrl Ping Message</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send Spybot Ctrl Ping Message</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_CTRL_PING_MESSAGE
	 * @model name="SendSpybotCtrlPingMessage"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_SPYBOT_CTRL_PING_MESSAGE_VALUE = 22;

	/**
	 * The '<em><b>Clear Timer</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Clear Timer</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CLEAR_TIMER
	 * @model name="ClearTimer"
	 * @generated
	 * @ordered
	 */
	public static final int CLEAR_TIMER_VALUE = 23;

	/**
	 * The '<em><b>Clear Counter</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Clear Counter</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CLEAR_COUNTER
	 * @model name="ClearCounter"
	 * @generated
	 * @ordered
	 */
	public static final int CLEAR_COUNTER_VALUE = 24;

	/**
	 * The '<em><b>Inc Counter</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Inc Counter</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #INC_COUNTER
	 * @model name="IncCounter"
	 * @generated
	 * @ordered
	 */
	public static final int INC_COUNTER_VALUE = 25;

	/**
	 * The '<em><b>Dec Counter</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Dec Counter</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #DEC_COUNTER
	 * @model name="DecCounter"
	 * @generated
	 * @ordered
	 */
	public static final int DEC_COUNTER_VALUE = 26;

	/**
	 * The '<em><b>Set Priority</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Priority</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_PRIORITY
	 * @model name="SetPriority"
	 * @generated
	 * @ordered
	 */
	public static final int SET_PRIORITY_VALUE = 27;

	/**
	 * The '<em><b>Event</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Event</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EVENT
	 * @model name="Event"
	 * @generated
	 * @ordered
	 */
	public static final int EVENT_VALUE = 28;

	/**
	 * The '<em><b>Create Datalog</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Create Datalog</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CREATE_DATALOG
	 * @model name="CreateDatalog"
	 * @generated
	 * @ordered
	 */
	public static final int CREATE_DATALOG_VALUE = 29;

	/**
	 * The '<em><b>Add To Datalog</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Add To Datalog</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ADD_TO_DATALOG
	 * @model name="AddToDatalog"
	 * @generated
	 * @ordered
	 */
	public static final int ADD_TO_DATALOG_VALUE = 30;

	/**
	 * The '<em><b>Wait</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Wait</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #WAIT
	 * @model name="Wait"
	 * @generated
	 * @ordered
	 */
	public static final int WAIT_VALUE = 31;

	/**
	 * The '<em><b>Set Random Seed</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Random Seed</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_RANDOM_SEED
	 * @model name="SetRandomSeed"
	 * @generated
	 * @ordered
	 */
	public static final int SET_RANDOM_SEED_VALUE = 32;

	/**
	 * The '<em><b>Set Sleep Time</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Sleep Time</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_SLEEP_TIME
	 * @model name="SetSleepTime"
	 * @generated
	 * @ordered
	 */
	public static final int SET_SLEEP_TIME_VALUE = 33;

	/**
	 * The '<em><b>Select Program</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Select Program</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SELECT_PROGRAM
	 * @model name="SelectProgram"
	 * @generated
	 * @ordered
	 */
	public static final int SELECT_PROGRAM_VALUE = 34;

	/**
	 * An array of all the '<em><b>Built In Unary Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final BuiltInUnaryFunctionEnum[] VALUES_ARRAY =
		new BuiltInUnaryFunctionEnum[] {
			CLEAR_SENSOR,
			ON,
			OFF,
			FLOAT,
			FWD,
			REV,
			TOGGLE,
			ON_FWD,
			ON_REV,
			ENABLE_OUTPUT,
			DISABLE_OUTPUT,
			INVERT_OUTPUT,
			OBVERT_OUTPUT,
			PLAY_SOUND,
			SELECT_DISPLAY,
			SEND_MESSAGE,
			SET_TX_POWER,
			SET_SERIAL_COMM,
			SET_SERIAL_PACKET,
			INIT_SPYBOT_COMM,
			SEND_SPYBOT_MSG,
			SET_SPYBOT_CTRL_PING_MESSAGE,
			SEND_SPYBOT_CTRL_PING_MESSAGE,
			CLEAR_TIMER,
			CLEAR_COUNTER,
			INC_COUNTER,
			DEC_COUNTER,
			SET_PRIORITY,
			EVENT,
			CREATE_DATALOG,
			ADD_TO_DATALOG,
			WAIT,
			SET_RANDOM_SEED,
			SET_SLEEP_TIME,
			SELECT_PROGRAM,
		};

	/**
	 * A public read-only list of all the '<em><b>Built In Unary Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<BuiltInUnaryFunctionEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Built In Unary Function Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInUnaryFunctionEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInUnaryFunctionEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Unary Function Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInUnaryFunctionEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInUnaryFunctionEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Unary Function Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInUnaryFunctionEnum get(int value) {
		switch (value) {
			case CLEAR_SENSOR_VALUE: return CLEAR_SENSOR;
			case ON_VALUE: return ON;
			case OFF_VALUE: return OFF;
			case FLOAT_VALUE: return FLOAT;
			case FWD_VALUE: return FWD;
			case REV_VALUE: return REV;
			case TOGGLE_VALUE: return TOGGLE;
			case ON_FWD_VALUE: return ON_FWD;
			case ON_REV_VALUE: return ON_REV;
			case ENABLE_OUTPUT_VALUE: return ENABLE_OUTPUT;
			case DISABLE_OUTPUT_VALUE: return DISABLE_OUTPUT;
			case INVERT_OUTPUT_VALUE: return INVERT_OUTPUT;
			case OBVERT_OUTPUT_VALUE: return OBVERT_OUTPUT;
			case PLAY_SOUND_VALUE: return PLAY_SOUND;
			case SELECT_DISPLAY_VALUE: return SELECT_DISPLAY;
			case SEND_MESSAGE_VALUE: return SEND_MESSAGE;
			case SET_TX_POWER_VALUE: return SET_TX_POWER;
			case SET_SERIAL_COMM_VALUE: return SET_SERIAL_COMM;
			case SET_SERIAL_PACKET_VALUE: return SET_SERIAL_PACKET;
			case INIT_SPYBOT_COMM_VALUE: return INIT_SPYBOT_COMM;
			case SEND_SPYBOT_MSG_VALUE: return SEND_SPYBOT_MSG;
			case SET_SPYBOT_CTRL_PING_MESSAGE_VALUE: return SET_SPYBOT_CTRL_PING_MESSAGE;
			case SEND_SPYBOT_CTRL_PING_MESSAGE_VALUE: return SEND_SPYBOT_CTRL_PING_MESSAGE;
			case CLEAR_TIMER_VALUE: return CLEAR_TIMER;
			case CLEAR_COUNTER_VALUE: return CLEAR_COUNTER;
			case INC_COUNTER_VALUE: return INC_COUNTER;
			case DEC_COUNTER_VALUE: return DEC_COUNTER;
			case SET_PRIORITY_VALUE: return SET_PRIORITY;
			case EVENT_VALUE: return EVENT;
			case CREATE_DATALOG_VALUE: return CREATE_DATALOG;
			case ADD_TO_DATALOG_VALUE: return ADD_TO_DATALOG;
			case WAIT_VALUE: return WAIT;
			case SET_RANDOM_SEED_VALUE: return SET_RANDOM_SEED;
			case SET_SLEEP_TIME_VALUE: return SET_SLEEP_TIME;
			case SELECT_PROGRAM_VALUE: return SELECT_PROGRAM;
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
	private BuiltInUnaryFunctionEnum(int value, String name, String literal) {
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
	
} //BuiltInUnaryFunctionEnum
