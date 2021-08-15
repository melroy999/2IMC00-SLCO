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
 * A representation of the literals of the enumeration '<em><b>Sensor Type Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getSensorTypeEnum()
 * @model
 * @generated
 */
public enum SensorTypeEnum implements Enumerator {
	/**
	 * The '<em><b>SENSOR TYPE NONE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_NONE_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_TYPE_NONE(0, "SENSOR_TYPE_NONE", "SENSOR_TYPE_NONE"),

	/**
	 * The '<em><b>SENSOR TYPE TOUCH</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_TOUCH_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_TYPE_TOUCH(1, "SENSOR_TYPE_TOUCH", "SENSOR_TYPE_TOUCH"),

	/**
	 * The '<em><b>SENSOR TYPE TEMPERATURE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_TEMPERATURE_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_TYPE_TEMPERATURE(2, "SENSOR_TYPE_TEMPERATURE", "SENSOR_TYPE_TEMPERATURE"),

	/**
	 * The '<em><b>SENSOR TYPE LIGHT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_LIGHT_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_TYPE_LIGHT(3, "SENSOR_TYPE_LIGHT", "SENSOR_TYPE_LIGHT"),

	/**
	 * The '<em><b>SENSOR TYPE ROTATION</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_ROTATION_VALUE
	 * @generated
	 * @ordered
	 */
	SENSOR_TYPE_ROTATION(4, "SENSOR_TYPE_ROTATION", "SENSOR_TYPE_ROTATION");

	/**
	 * The '<em><b>SENSOR TYPE NONE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR TYPE NONE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_NONE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_TYPE_NONE_VALUE = 0;

	/**
	 * The '<em><b>SENSOR TYPE TOUCH</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR TYPE TOUCH</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_TOUCH
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_TYPE_TOUCH_VALUE = 1;

	/**
	 * The '<em><b>SENSOR TYPE TEMPERATURE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR TYPE TEMPERATURE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_TEMPERATURE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_TYPE_TEMPERATURE_VALUE = 2;

	/**
	 * The '<em><b>SENSOR TYPE LIGHT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR TYPE LIGHT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_LIGHT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_TYPE_LIGHT_VALUE = 3;

	/**
	 * The '<em><b>SENSOR TYPE ROTATION</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SENSOR TYPE ROTATION</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SENSOR_TYPE_ROTATION
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SENSOR_TYPE_ROTATION_VALUE = 4;

	/**
	 * An array of all the '<em><b>Sensor Type Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final SensorTypeEnum[] VALUES_ARRAY =
		new SensorTypeEnum[] {
			SENSOR_TYPE_NONE,
			SENSOR_TYPE_TOUCH,
			SENSOR_TYPE_TEMPERATURE,
			SENSOR_TYPE_LIGHT,
			SENSOR_TYPE_ROTATION,
		};

	/**
	 * A public read-only list of all the '<em><b>Sensor Type Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<SensorTypeEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Sensor Type Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorTypeEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SensorTypeEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Sensor Type Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorTypeEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SensorTypeEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Sensor Type Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SensorTypeEnum get(int value) {
		switch (value) {
			case SENSOR_TYPE_NONE_VALUE: return SENSOR_TYPE_NONE;
			case SENSOR_TYPE_TOUCH_VALUE: return SENSOR_TYPE_TOUCH;
			case SENSOR_TYPE_TEMPERATURE_VALUE: return SENSOR_TYPE_TEMPERATURE;
			case SENSOR_TYPE_LIGHT_VALUE: return SENSOR_TYPE_LIGHT;
			case SENSOR_TYPE_ROTATION_VALUE: return SENSOR_TYPE_ROTATION;
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
	private SensorTypeEnum(int value, String name, String literal) {
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
	
} //SensorTypeEnum
