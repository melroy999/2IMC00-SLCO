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
 * A representation of the literals of the enumeration '<em><b>Serial Comm Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getSerialCommEnum()
 * @model
 * @generated
 */
public enum SerialCommEnum implements Enumerator {
	/**
	 * The '<em><b>SERIAL COMM DEFAULT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_COMM_DEFAULT_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_COMM_DEFAULT(0, "SERIAL_COMM_DEFAULT", "SERIAL_COMM_DEFAULT"),

	/**
	 * The '<em><b>SERIAL COMM 4800</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_COMM_4800_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_COMM_4800(1, "SERIAL_COMM_4800", "SERIAL_COMM_4800"),

	/**
	 * The '<em><b>SERIAL COMM DUTY25</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_COMM_DUTY25_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_COMM_DUTY25(2, "SERIAL_COMM_DUTY25", "SERIAL_COMM_DUTY25"),

	/**
	 * The '<em><b>SERIAL COMM 76KHZ</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_COMM_76KHZ_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_COMM_76KHZ(3, "SERIAL_COMM_76KHZ", "SERIAL_COMM_76KHZ");

	/**
	 * The '<em><b>SERIAL COMM DEFAULT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL COMM DEFAULT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_COMM_DEFAULT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_COMM_DEFAULT_VALUE = 0;

	/**
	 * The '<em><b>SERIAL COMM 4800</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL COMM 4800</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_COMM_4800
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_COMM_4800_VALUE = 1;

	/**
	 * The '<em><b>SERIAL COMM DUTY25</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL COMM DUTY25</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_COMM_DUTY25
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_COMM_DUTY25_VALUE = 2;

	/**
	 * The '<em><b>SERIAL COMM 76KHZ</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL COMM 76KHZ</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_COMM_76KHZ
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_COMM_76KHZ_VALUE = 3;

	/**
	 * An array of all the '<em><b>Serial Comm Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final SerialCommEnum[] VALUES_ARRAY =
		new SerialCommEnum[] {
			SERIAL_COMM_DEFAULT,
			SERIAL_COMM_4800,
			SERIAL_COMM_DUTY25,
			SERIAL_COMM_76KHZ,
		};

	/**
	 * A public read-only list of all the '<em><b>Serial Comm Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<SerialCommEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Serial Comm Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialCommEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SerialCommEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Serial Comm Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialCommEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SerialCommEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Serial Comm Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialCommEnum get(int value) {
		switch (value) {
			case SERIAL_COMM_DEFAULT_VALUE: return SERIAL_COMM_DEFAULT;
			case SERIAL_COMM_4800_VALUE: return SERIAL_COMM_4800;
			case SERIAL_COMM_DUTY25_VALUE: return SERIAL_COMM_DUTY25;
			case SERIAL_COMM_76KHZ_VALUE: return SERIAL_COMM_76KHZ;
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
	private SerialCommEnum(int value, String name, String literal) {
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
	
} //SerialCommEnum
