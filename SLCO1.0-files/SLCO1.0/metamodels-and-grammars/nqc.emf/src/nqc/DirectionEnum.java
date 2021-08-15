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
 * A representation of the literals of the enumeration '<em><b>Direction Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getDirectionEnum()
 * @model
 * @generated
 */
public enum DirectionEnum implements Enumerator {
	/**
	 * The '<em><b>OUT FWD</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OUT_FWD_VALUE
	 * @generated
	 * @ordered
	 */
	OUT_FWD(0, "OUT_FWD", "OUT_FWD"),

	/**
	 * The '<em><b>OUT REV</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OUT_REV_VALUE
	 * @generated
	 * @ordered
	 */
	OUT_REV(1, "OUT_REV", "OUT_REV"),

	/**
	 * The '<em><b>OUT TOGGLE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OUT_TOGGLE_VALUE
	 * @generated
	 * @ordered
	 */
	OUT_TOGGLE(2, "OUT_TOGGLE", "OUT_TOGGLE");

	/**
	 * The '<em><b>OUT FWD</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>OUT FWD</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OUT_FWD
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int OUT_FWD_VALUE = 0;

	/**
	 * The '<em><b>OUT REV</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>OUT REV</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OUT_REV
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int OUT_REV_VALUE = 1;

	/**
	 * The '<em><b>OUT TOGGLE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>OUT TOGGLE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OUT_TOGGLE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int OUT_TOGGLE_VALUE = 2;

	/**
	 * An array of all the '<em><b>Direction Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final DirectionEnum[] VALUES_ARRAY =
		new DirectionEnum[] {
			OUT_FWD,
			OUT_REV,
			OUT_TOGGLE,
		};

	/**
	 * A public read-only list of all the '<em><b>Direction Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<DirectionEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Direction Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static DirectionEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			DirectionEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Direction Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static DirectionEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			DirectionEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Direction Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static DirectionEnum get(int value) {
		switch (value) {
			case OUT_FWD_VALUE: return OUT_FWD;
			case OUT_REV_VALUE: return OUT_REV;
			case OUT_TOGGLE_VALUE: return OUT_TOGGLE;
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
	private DirectionEnum(int value, String name, String literal) {
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
	
} //DirectionEnum
