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
 * A representation of the literals of the enumeration '<em><b>Sensor Config Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getSensorConfigEnum()
 * @model
 * @generated
 */
public enum SensorConfigEnum implements Enumerator {
	/**
	 * The '<em><b>SENSOR TOUCH</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TOUCH_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_TOUCH(0, "SENSOR_TOUCH", "SENSOR_TOUCH"),

	/**
	 * The '<em><b>SENSOR LIGHT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_LIGHT_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_LIGHT(1, "SENSOR_LIGHT", "SENSOR_LIGHT"),

	/**
	 * The '<em><b>SENSOR ROTATION</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_ROTATION_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_ROTATION(2, "SENSOR_ROTATION", "SENSOR_ROTATION"),

	/**
	 * The '<em><b>SENSOR CELSIUS</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_CELSIUS_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_CELSIUS(3, "SENSOR_CELSIUS", "SENSOR_CELSIUS"),

	/**
	 * The '<em><b>SENSOR FAHRENHEIT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_FAHRENHEIT_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_FAHRENHEIT(4, "SENSOR_FAHRENHEIT", "SENSOR_FAHRENHEIT"),

	/**
	 * The '<em><b>SENSOR PULSE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_PULSE_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_PULSE(5, "SENSOR_PULSE", "SENSOR_PULSE"),

	/**
	 * The '<em><b>SENSOR EDGE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_EDGE_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_EDGE(6, "SENSOR_EDGE", "SENSOR_EDGE");

	/**
	 * The '<em><b>SENSOR TOUCH</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR TOUCH</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TOUCH
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_TOUCH_VALUE = 0;

	/**
	 * The '<em><b>SENSOR LIGHT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR LIGHT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_LIGHT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_LIGHT_VALUE = 1;

	/**
	 * The '<em><b>SENSOR ROTATION</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR ROTATION</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_ROTATION
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_ROTATION_VALUE = 2;

	/**
	 * The '<em><b>SENSOR CELSIUS</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR CELSIUS</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_CELSIUS
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_CELSIUS_VALUE = 3;

	/**
	 * The '<em><b>SENSOR FAHRENHEIT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR FAHRENHEIT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_FAHRENHEIT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_FAHRENHEIT_VALUE = 4;

	/**
	 * The '<em><b>SENSOR PULSE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR PULSE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_PULSE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_PULSE_VALUE = 5;

	/**
	 * The '<em><b>SENSOR EDGE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR EDGE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_EDGE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_EDGE_VALUE = 6;

	/**
	 * An array of all the '<em><b>Sensor Config Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final SensorConfigEnum[] VALUES_ARRAY =
		new SensorConfigEnum[] {
			SENSOR_TOUCH,
			SENSOR_LIGHT,
			SENSOR_ROTATION,
			SENSOR_CELSIUS,
			SENSOR_FAHRENHEIT,
			SENSOR_PULSE,
			SENSOR_EDGE,
		};

	/**
	 * A public read-only list of all the '<em><b>Sensor Config Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<SensorConfigEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Sensor Config Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorConfigEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SensorConfigEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Sensor Config Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorConfigEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SensorConfigEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Sensor Config Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorConfigEnum get(int value) {
		switch (value) {
			case SENSOR_TOUCH_VALUE: return SENSOR_TOUCH;
			case SENSOR_LIGHT_VALUE: return SENSOR_LIGHT;
			case SENSOR_ROTATION_VALUE: return SENSOR_ROTATION;
			case SENSOR_CELSIUS_VALUE: return SENSOR_CELSIUS;
			case SENSOR_FAHRENHEIT_VALUE: return SENSOR_FAHRENHEIT;
			case SENSOR_PULSE_VALUE: return SENSOR_PULSE;
			case SENSOR_EDGE_VALUE: return SENSOR_EDGE;
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
	private SensorConfigEnum(int value, String name, String literal) {
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
	
} //SensorConfigEnum
