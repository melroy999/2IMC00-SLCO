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
 * A representation of the literals of the enumeration '<em><b>Serial Channel Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getSerialChannelEnum()
 * @model
 * @generated
 */
public enum SerialChannelEnum implements Enumerator {
	/**
	 * The '<em><b>SERIAL CHANNEL IR</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_CHANNEL_IR_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_CHANNEL_IR(0, "SERIAL_CHANNEL_IR", "SERIAL_CHANNEL_IR"),

	/**
	 * The '<em><b>SERIAL CHANNEL PC</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_CHANNEL_PC_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_CHANNEL_PC(1, "SERIAL_CHANNEL_PC", "SERIAL_CHANNEL_PC");

	/**
	 * The '<em><b>SERIAL CHANNEL IR</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL CHANNEL IR</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_CHANNEL_IR
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_CHANNEL_IR_VALUE = 0;

	/**
	 * The '<em><b>SERIAL CHANNEL PC</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL CHANNEL PC</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_CHANNEL_PC
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_CHANNEL_PC_VALUE = 1;

	/**
	 * An array of all the '<em><b>Serial Channel Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final SerialChannelEnum[] VALUES_ARRAY =
		new SerialChannelEnum[] {
			SERIAL_CHANNEL_IR,
			SERIAL_CHANNEL_PC,
		};

	/**
	 * A public read-only list of all the '<em><b>Serial Channel Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<SerialChannelEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Serial Channel Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialChannelEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SerialChannelEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Serial Channel Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialChannelEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SerialChannelEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Serial Channel Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialChannelEnum get(int value) {
		switch (value) {
			case SERIAL_CHANNEL_IR_VALUE: return SERIAL_CHANNEL_IR;
			case SERIAL_CHANNEL_PC_VALUE: return SERIAL_CHANNEL_PC;
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
	private SerialChannelEnum(int value, String name, String literal) {
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
	
} //SerialChannelEnum
