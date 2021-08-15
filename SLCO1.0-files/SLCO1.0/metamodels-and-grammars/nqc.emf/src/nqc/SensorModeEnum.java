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
 * A representation of the literals of the enumeration '<em><b>Sensor Mode Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getSensorModeEnum()
 * @model
 * @generated
 */
public enum SensorModeEnum implements Enumerator {
	/**
	 * The '<em><b>SENSOR MODE RAW</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_RAW_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_MODE_RAW(0, "SENSOR_MODE_RAW", "SENSOR_MODE_RAW"),

	/**
	 * The '<em><b>SENSOR MODE BOOL</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_BOOL_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_MODE_BOOL(1, "SENSOR_MODE_BOOL", "SENSOR_MODE_BOOL"),

	/**
	 * The '<em><b>SENSOR MODE EDGE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_EDGE_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_MODE_EDGE(2, "SENSOR_MODE_EDGE", "SENSOR_MODE_EDGE"),

	/**
	 * The '<em><b>SENSOR MODE PULSE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_PULSE_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_MODE_PULSE(3, "SENSOR_MODE_PULSE", "SENSOR_MODE_PULSE"),

	/**
	 * The '<em><b>SENSOR MODE PERCENT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_PERCENT_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_MODE_PERCENT(4, "SENSOR_MODE_PERCENT", "SENSOR_MODE_PERCENT"),

	/**
	 * The '<em><b>SENSOR MODE FAHRENHEIT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_FAHRENHEIT_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_MODE_FAHRENHEIT(5, "SENSOR_MODE_FAHRENHEIT", "SENSOR_MODE_FAHRENHEIT"),

	/**
	 * The '<em><b>SENSOR MODE CELSIUS</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_CELSIUS_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_MODE_CELSIUS(6, "SENSOR_MODE_CELSIUS", "SENSOR_MODE_CELSIUS"),

	/**
	 * The '<em><b>SENSOR MODE ROTATION</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_ROTATION_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_MODE_ROTATION(7, "SENSOR_MODE_ROTATION", "SENSOR_MODE_ROTATION");

	/**
	 * The '<em><b>SENSOR MODE RAW</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR MODE RAW</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_RAW
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_MODE_RAW_VALUE = 0;

	/**
	 * The '<em><b>SENSOR MODE BOOL</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR MODE BOOL</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_BOOL
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_MODE_BOOL_VALUE = 1;

	/**
	 * The '<em><b>SENSOR MODE EDGE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR MODE EDGE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_EDGE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_MODE_EDGE_VALUE = 2;

	/**
	 * The '<em><b>SENSOR MODE PULSE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR MODE PULSE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_PULSE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_MODE_PULSE_VALUE = 3;

	/**
	 * The '<em><b>SENSOR MODE PERCENT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR MODE PERCENT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_PERCENT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_MODE_PERCENT_VALUE = 4;

	/**
	 * The '<em><b>SENSOR MODE FAHRENHEIT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR MODE FAHRENHEIT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_FAHRENHEIT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_MODE_FAHRENHEIT_VALUE = 5;

	/**
	 * The '<em><b>SENSOR MODE CELSIUS</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR MODE CELSIUS</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_CELSIUS
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_MODE_CELSIUS_VALUE = 6;

	/**
	 * The '<em><b>SENSOR MODE ROTATION</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR MODE ROTATION</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_MODE_ROTATION
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_MODE_ROTATION_VALUE = 7;

	/**
	 * An array of all the '<em><b>Sensor Mode Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final SensorModeEnum[] VALUES_ARRAY =
		new SensorModeEnum[] {
			SENSOR_MODE_RAW,
			SENSOR_MODE_BOOL,
			SENSOR_MODE_EDGE,
			SENSOR_MODE_PULSE,
			SENSOR_MODE_PERCENT,
			SENSOR_MODE_FAHRENHEIT,
			SENSOR_MODE_CELSIUS,
			SENSOR_MODE_ROTATION,
		};

	/**
	 * A public read-only list of all the '<em><b>Sensor Mode Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<SensorModeEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Sensor Mode Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorModeEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SensorModeEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Sensor Mode Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorModeEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SensorModeEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Sensor Mode Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorModeEnum get(int value) {
		switch (value) {
			case SENSOR_MODE_RAW_VALUE: return SENSOR_MODE_RAW;
			case SENSOR_MODE_BOOL_VALUE: return SENSOR_MODE_BOOL;
			case SENSOR_MODE_EDGE_VALUE: return SENSOR_MODE_EDGE;
			case SENSOR_MODE_PULSE_VALUE: return SENSOR_MODE_PULSE;
			case SENSOR_MODE_PERCENT_VALUE: return SENSOR_MODE_PERCENT;
			case SENSOR_MODE_FAHRENHEIT_VALUE: return SENSOR_MODE_FAHRENHEIT;
			case SENSOR_MODE_CELSIUS_VALUE: return SENSOR_MODE_CELSIUS;
			case SENSOR_MODE_ROTATION_VALUE: return SENSOR_MODE_ROTATION;
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
	private SensorModeEnum(int value, String name, String literal) {
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
	
} //SensorModeEnum
