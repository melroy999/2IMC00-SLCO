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
 * A representation of the literals of the enumeration '<em><b>Serial Packet Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getSerialPacketEnum()
 * @model
 * @generated
 */
public enum SerialPacketEnum implements Enumerator {
	/**
	 * The '<em><b>SERIAL PACKET DEFAULT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_PACKET_DEFAULT_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_PACKET_DEFAULT(0, "SERIAL_PACKET_DEFAULT", "SERIAL_PACKET_DEFAULT"),

	/**
	 * The '<em><b>SERIAL PACKET PREAMBLE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_PACKET_PREAMBLE_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_PACKET_PREAMBLE(1, "SERIAL_PACKET_PREAMBLE", "SERIAL_PACKET_PREAMBLE"),

	/**
	 * The '<em><b>SERIAL PACKET NEGATED</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_PACKET_NEGATED_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_PACKET_NEGATED(2, "SERIAL_PACKET_NEGATED", "SERIAL_PACKET_NEGATED"),

	/**
	 * The '<em><b>SERIAL PACKET CHECKSUM</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_PACKET_CHECKSUM_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_PACKET_CHECKSUM(3, "SERIAL_PACKET_CHECKSUM", "SERIAL_PACKET_CHECKSUM"),

	/**
	 * The '<em><b>SERIAL PACKET RCX</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_PACKET_RCX_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_PACKET_RCX(4, "SERIAL_PACKET_RCX", "SERIAL_PACKET_RCX");

	/**
	 * The '<em><b>SERIAL PACKET DEFAULT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL PACKET DEFAULT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_PACKET_DEFAULT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_PACKET_DEFAULT_VALUE = 0;

	/**
	 * The '<em><b>SERIAL PACKET PREAMBLE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL PACKET PREAMBLE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_PACKET_PREAMBLE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_PACKET_PREAMBLE_VALUE = 1;

	/**
	 * The '<em><b>SERIAL PACKET NEGATED</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL PACKET NEGATED</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_PACKET_NEGATED
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_PACKET_NEGATED_VALUE = 2;

	/**
	 * The '<em><b>SERIAL PACKET CHECKSUM</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL PACKET CHECKSUM</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_PACKET_CHECKSUM
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_PACKET_CHECKSUM_VALUE = 3;

	/**
	 * The '<em><b>SERIAL PACKET RCX</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL PACKET RCX</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_PACKET_RCX
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_PACKET_RCX_VALUE = 4;

	/**
	 * An array of all the '<em><b>Serial Packet Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final SerialPacketEnum[] VALUES_ARRAY =
		new SerialPacketEnum[] {
			SERIAL_PACKET_DEFAULT,
			SERIAL_PACKET_PREAMBLE,
			SERIAL_PACKET_NEGATED,
			SERIAL_PACKET_CHECKSUM,
			SERIAL_PACKET_RCX,
		};

	/**
	 * A public read-only list of all the '<em><b>Serial Packet Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<SerialPacketEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Serial Packet Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialPacketEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SerialPacketEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Serial Packet Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialPacketEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SerialPacketEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Serial Packet Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialPacketEnum get(int value) {
		switch (value) {
			case SERIAL_PACKET_DEFAULT_VALUE: return SERIAL_PACKET_DEFAULT;
			case SERIAL_PACKET_PREAMBLE_VALUE: return SERIAL_PACKET_PREAMBLE;
			case SERIAL_PACKET_NEGATED_VALUE: return SERIAL_PACKET_NEGATED;
			case SERIAL_PACKET_CHECKSUM_VALUE: return SERIAL_PACKET_CHECKSUM;
			case SERIAL_PACKET_RCX_VALUE: return SERIAL_PACKET_RCX;
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
	private SerialPacketEnum(int value, String name, String literal) {
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
	
} //SerialPacketEnum
