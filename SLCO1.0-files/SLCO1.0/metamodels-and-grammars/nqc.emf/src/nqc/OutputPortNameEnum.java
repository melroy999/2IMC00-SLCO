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
 * A representation of the literals of the enumeration '<em><b>Output Port Name Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getOutputPortNameEnum()
 * @model
 * @generated
 */
public enum OutputPortNameEnum implements Enumerator {
	/**
	 * The '<em><b>OUT A</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OUT_A_VALUE
	 * @generated
	 * @ordered
	 */
	OUT_A(0, "OUT_A", "OUT_A"),

	/**
	 * The '<em><b>OUT B</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OUT_B_VALUE
	 * @generated
	 * @ordered
	 */
	OUT_B(1, "OUT_B", "OUT_B"),

	/**
	 * The '<em><b>OUT C</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OUT_C_VALUE
	 * @generated
	 * @ordered
	 */
	OUT_C(2, "OUT_C", "OUT_C");

	/**
	 * The '<em><b>OUT A</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>OUT A</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OUT_A
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int OUT_A_VALUE = 0;

	/**
	 * The '<em><b>OUT B</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>OUT B</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OUT_B
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int OUT_B_VALUE = 1;

	/**
	 * The '<em><b>OUT C</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>OUT C</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OUT_C
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int OUT_C_VALUE = 2;

	/**
	 * An array of all the '<em><b>Output Port Name Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final OutputPortNameEnum[] VALUES_ARRAY =
		new OutputPortNameEnum[] {
			OUT_A,
			OUT_B,
			OUT_C,
		};

	/**
	 * A public read-only list of all the '<em><b>Output Port Name Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<OutputPortNameEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Output Port Name Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static OutputPortNameEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			OutputPortNameEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Output Port Name Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static OutputPortNameEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			OutputPortNameEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Output Port Name Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static OutputPortNameEnum get(int value) {
		switch (value) {
			case OUT_A_VALUE: return OUT_A;
			case OUT_B_VALUE: return OUT_B;
			case OUT_C_VALUE: return OUT_C;
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
	private OutputPortNameEnum(int value, String name, String literal) {
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
	
} //OutputPortNameEnum
