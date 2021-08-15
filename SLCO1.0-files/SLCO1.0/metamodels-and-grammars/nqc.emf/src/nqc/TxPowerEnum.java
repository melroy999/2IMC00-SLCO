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
 * A representation of the literals of the enumeration '<em><b>Tx Power Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getTxPowerEnum()
 * @model
 * @generated
 */
public enum TxPowerEnum implements Enumerator {
	/**
	 * The '<em><b>TX POWER LO</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #TX_POWER_LO_VALUE
	 * @generated
	 * @ordered
	 */
	TX_POWER_LO(0, "TX_POWER_LO", "TX_POWER_LO"),

	/**
	 * The '<em><b>TX POWER HIGH</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #TX_POWER_HIGH_VALUE
	 * @generated
	 * @ordered
	 */
	TX_POWER_HIGH(0, "TX_POWER_HIGH", "TX_POWER_HIGH");

	/**
	 * The '<em><b>TX POWER LO</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>TX POWER LO</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #TX_POWER_LO
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int TX_POWER_LO_VALUE = 0;

	/**
	 * The '<em><b>TX POWER HIGH</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>TX POWER HIGH</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #TX_POWER_HIGH
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int TX_POWER_HIGH_VALUE = 0;

	/**
	 * An array of all the '<em><b>Tx Power Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final TxPowerEnum[] VALUES_ARRAY =
		new TxPowerEnum[] {
			TX_POWER_LO,
			TX_POWER_HIGH,
		};

	/**
	 * A public read-only list of all the '<em><b>Tx Power Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<TxPowerEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Tx Power Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static TxPowerEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			TxPowerEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Tx Power Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static TxPowerEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			TxPowerEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Tx Power Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static TxPowerEnum get(int value) {
		switch (value) {
			case TX_POWER_LO_VALUE: return TX_POWER_LO;
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
	private TxPowerEnum(int value, String name, String literal) {
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
	
} //TxPowerEnum
