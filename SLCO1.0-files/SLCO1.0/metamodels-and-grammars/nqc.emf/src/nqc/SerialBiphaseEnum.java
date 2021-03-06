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
 * A representation of the literals of the enumeration '<em><b>Serial Biphase Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getSerialBiphaseEnum()
 * @model
 * @generated
 */
public enum SerialBiphaseEnum implements Enumerator {
	/**
	 * The '<em><b>SERIAL BIPHASE OFF</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_BIPHASE_OFF_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_BIPHASE_OFF(0, "SERIAL_BIPHASE_OFF", "SERIAL_BIPHASE_OFF"),

	/**
	 * The '<em><b>SERIAL BIPHASE ON</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SERIAL_BIPHASE_ON_VALUE
	 * @generated
	 * @ordered
	 */
	SERIAL_BIPHASE_ON(1, "SERIAL_BIPHASE_ON", "SERIAL_BIPHASE_ON");

	/**
	 * The '<em><b>SERIAL BIPHASE OFF</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL BIPHASE OFF</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_BIPHASE_OFF
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_BIPHASE_OFF_VALUE = 0;

	/**
	 * The '<em><b>SERIAL BIPHASE ON</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SERIAL BIPHASE ON</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SERIAL_BIPHASE_ON
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SERIAL_BIPHASE_ON_VALUE = 1;

	/**
	 * An array of all the '<em><b>Serial Biphase Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final SerialBiphaseEnum[] VALUES_ARRAY =
		new SerialBiphaseEnum[] {
			SERIAL_BIPHASE_OFF,
			SERIAL_BIPHASE_ON,
		};

	/**
	 * A public read-only list of all the '<em><b>Serial Biphase Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<SerialBiphaseEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Serial Biphase Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialBiphaseEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SerialBiphaseEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Serial Biphase Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialBiphaseEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SerialBiphaseEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Serial Biphase Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SerialBiphaseEnum get(int value) {
		switch (value) {
			case SERIAL_BIPHASE_OFF_VALUE: return SERIAL_BIPHASE_OFF;
			case SERIAL_BIPHASE_ON_VALUE: return SERIAL_BIPHASE_ON;
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
	private SerialBiphaseEnum(int value, String name, String literal) {
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
	
} //SerialBiphaseEnum
