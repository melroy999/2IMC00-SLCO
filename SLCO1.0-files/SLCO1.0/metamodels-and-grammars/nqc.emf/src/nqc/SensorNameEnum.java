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
 * A representation of the literals of the enumeration '<em><b>Sensor Name Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getSensorNameEnum()
 * @model
 * @generated
 */
public enum SensorNameEnum implements Enumerator {
	/**
	 * The '<em><b>SENSOR 1</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_1_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_1(0, "SENSOR_1", "SENSOR_1"),

	/**
	 * The '<em><b>SENSOR 2</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_2_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_2(1, "SENSOR_2", "SENSOR_2"),

	/**
	 * The '<em><b>SENSOR 3</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_3_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_3(2, "SENSOR_3", "SENSOR_3");

	/**
	 * The '<em><b>SENSOR 1</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR 1</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_1
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_1_VALUE = 0;

	/**
	 * The '<em><b>SENSOR 2</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR 2</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_2
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_2_VALUE = 1;

	/**
	 * The '<em><b>SENSOR 3</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR 3</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_3
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_3_VALUE = 2;

	/**
	 * An array of all the '<em><b>Sensor Name Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final SensorNameEnum[] VALUES_ARRAY =
		new SensorNameEnum[] {
			SENSOR_1,
			SENSOR_2,
			SENSOR_3,
		};

	/**
	 * A public read-only list of all the '<em><b>Sensor Name Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<SensorNameEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Sensor Name Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorNameEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SensorNameEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Sensor Name Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorNameEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SensorNameEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Sensor Name Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorNameEnum get(int value) {
		switch (value) {
			case SENSOR_1_VALUE: return SENSOR_1;
			case SENSOR_2_VALUE: return SENSOR_2;
			case SENSOR_3_VALUE: return SENSOR_3;
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
	private SensorNameEnum(int value, String name, String literal) {
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
	
} //SensorNameEnum
