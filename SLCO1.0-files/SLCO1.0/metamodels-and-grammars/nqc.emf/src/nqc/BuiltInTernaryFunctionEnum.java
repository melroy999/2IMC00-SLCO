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
 * A representation of the literals of the enumeration '<em><b>Built In Ternary Function Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getBuiltInTernaryFunctionEnum()
 * @model
 * @generated
 */
public enum BuiltInTernaryFunctionEnum implements Enumerator {
	/**
	 * The '<em><b>Set Spybot Ping</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_SPYBOT_PING_VALUE
	 * @generated
	 * @ordered
	 */
	SET_SPYBOT_PING(0, "SetSpybotPing", "SetSpybotPing"),

	/**
	 * The '<em><b>Send Spybot Ping</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_PING_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_SPYBOT_PING(1, "SendSpybotPing", "SendSpybotPing"),

	/**
	 * The '<em><b>Set RC Message</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_RC_MESSAGE_VALUE
	 * @generated
	 * @ordered
	 */
	SET_RC_MESSAGE(2, "SetRCMessage", "SetRCMessage"),

	/**
	 * The '<em><b>Send RC Message</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SEND_RC_MESSAGE_VALUE
	 * @generated
	 * @ordered
	 */
	SEND_RC_MESSAGE(3, "SendRCMessage", "SendRCMessage"),

	/**
	 * The '<em><b>Set Event</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SET_EVENT_VALUE
	 * @generated
	 * @ordered
	 */
	SET_EVENT(4, "SetEvent", "SetEvent");

	/**
	 * The '<em><b>Set Spybot Ping</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Spybot Ping</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_SPYBOT_PING
	 * @model name="SetSpybotPing"
	 * @generated
	 * @ordered
	 */
	public static final int SET_SPYBOT_PING_VALUE = 0;

	/**
	 * The '<em><b>Send Spybot Ping</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send Spybot Ping</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_SPYBOT_PING
	 * @model name="SendSpybotPing"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_SPYBOT_PING_VALUE = 1;

	/**
	 * The '<em><b>Set RC Message</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set RC Message</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_RC_MESSAGE
	 * @model name="SetRCMessage"
	 * @generated
	 * @ordered
	 */
	public static final int SET_RC_MESSAGE_VALUE = 2;

	/**
	 * The '<em><b>Send RC Message</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Send RC Message</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SEND_RC_MESSAGE
	 * @model name="SendRCMessage"
	 * @generated
	 * @ordered
	 */
	public static final int SEND_RC_MESSAGE_VALUE = 3;

	/**
	 * The '<em><b>Set Event</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Set Event</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SET_EVENT
	 * @model name="SetEvent"
	 * @generated
	 * @ordered
	 */
	public static final int SET_EVENT_VALUE = 4;

	/**
	 * An array of all the '<em><b>Built In Ternary Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final BuiltInTernaryFunctionEnum[] VALUES_ARRAY =
		new BuiltInTernaryFunctionEnum[] {
			SET_SPYBOT_PING,
			SEND_SPYBOT_PING,
			SET_RC_MESSAGE,
			SEND_RC_MESSAGE,
			SET_EVENT,
		};

	/**
	 * A public read-only list of all the '<em><b>Built In Ternary Function Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<BuiltInTernaryFunctionEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Built In Ternary Function Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInTernaryFunctionEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInTernaryFunctionEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Ternary Function Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInTernaryFunctionEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			BuiltInTernaryFunctionEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Built In Ternary Function Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static BuiltInTernaryFunctionEnum get(int value) {
		switch (value) {
			case SET_SPYBOT_PING_VALUE: return SET_SPYBOT_PING;
			case SEND_SPYBOT_PING_VALUE: return SEND_SPYBOT_PING;
			case SET_RC_MESSAGE_VALUE: return SET_RC_MESSAGE;
			case SEND_RC_MESSAGE_VALUE: return SEND_RC_MESSAGE;
			case SET_EVENT_VALUE: return SET_EVENT;
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
	private BuiltInTernaryFunctionEnum(int value, String name, String literal) {
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
	
} //BuiltInTernaryFunctionEnum
