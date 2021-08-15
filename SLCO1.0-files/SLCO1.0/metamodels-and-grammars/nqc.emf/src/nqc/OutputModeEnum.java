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
 * A representation of the literals of the enumeration '<em><b>Output Mode Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getOutputModeEnum()
 * @model
 * @generated
 */
public enum OutputModeEnum implements Enumerator {
	/**
	 * The '<em><b>OUT OFF</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OUT_OFF_VALUE
	 * @generated
	 * @ordered
	 */
	OUT_OFF(0, "OUT_OFF", "OUT_OFF"),

	/**
	 * The '<em><b>OUT ON</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OUT_ON_VALUE
	 * @generated
	 * @ordered
	 */
	OUT_ON(1, "OUT_ON", "OUT_ON"),

	/**
	 * The '<em><b>OUT FLOAT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OUT_FLOAT_VALUE
	 * @generated
	 * @ordered
	 */
	OUT_FLOAT(2, "OUT_FLOAT", "OUT_FLOAT");

	/**
	 * The '<em><b>OUT OFF</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>OUT OFF</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OUT_OFF
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int OUT_OFF_VALUE = 0;

	/**
	 * The '<em><b>OUT ON</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>OUT ON</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OUT_ON
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int OUT_ON_VALUE = 1;

	/**
	 * The '<em><b>OUT FLOAT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>OUT FLOAT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OUT_FLOAT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int OUT_FLOAT_VALUE = 2;

	/**
	 * An array of all the '<em><b>Output Mode Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final OutputModeEnum[] VALUES_ARRAY =
		new OutputModeEnum[] {
			OUT_OFF,
			OUT_ON,
			OUT_FLOAT,
		};

	/**
	 * A public read-only list of all the '<em><b>Output Mode Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<OutputModeEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Output Mode Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static OutputModeEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			OutputModeEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Output Mode Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static OutputModeEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			OutputModeEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Output Mode Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static OutputModeEnum get(int value) {
		switch (value) {
			case OUT_OFF_VALUE: return OUT_OFF;
			case OUT_ON_VALUE: return OUT_ON;
			case OUT_FLOAT_VALUE: return OUT_FLOAT;
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
	private OutputModeEnum(int value, String name, String literal) {
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
	
} //OutputModeEnum
