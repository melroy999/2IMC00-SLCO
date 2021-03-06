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
 * A representation of the literals of the enumeration '<em><b>Sound Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getSoundEnum()
 * @model
 * @generated
 */
public enum SoundEnum implements Enumerator {
	/**
	 * The '<em><b>SOUND CLICK</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SOUND_CLICK_VALUE
	 * @generated
	 * @ordered
	 */
	SOUND_CLICK(0, "SOUND_CLICK", "SOUND_CLICK"),

	/**
	 * The '<em><b>SOUND DOUBLE BEEP</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SOUND_DOUBLE_BEEP_VALUE
	 * @generated
	 * @ordered
	 */
	SOUND_DOUBLE_BEEP(1, "SOUND_DOUBLE_BEEP", "SOUND_DOUBLE_BEEP"),

	/**
	 * The '<em><b>SOUND DOWN</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SOUND_DOWN_VALUE
	 * @generated
	 * @ordered
	 */
	SOUND_DOWN(2, "SOUND_DOWN", "SOUND_DOWN"),

	/**
	 * The '<em><b>SOUND UP</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SOUND_UP_VALUE
	 * @generated
	 * @ordered
	 */
	SOUND_UP(3, "SOUND_UP", "SOUND_UP"),

	/**
	 * The '<em><b>SOUND LOW BEEP</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SOUND_LOW_BEEP_VALUE
	 * @generated
	 * @ordered
	 */
	SOUND_LOW_BEEP(4, "SOUND_LOW_BEEP", "SOUND_LOW_BEEP"),

	/**
	 * The '<em><b>SOUND FAST UP</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SOUND_FAST_UP_VALUE
	 * @generated
	 * @ordered
	 */
	SOUND_FAST_UP(5, "SOUND_FAST_UP", "SOUND_FAST_UP");

	/**
	 * The '<em><b>SOUND CLICK</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SOUND CLICK</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SOUND_CLICK
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SOUND_CLICK_VALUE = 0;

	/**
	 * The '<em><b>SOUND DOUBLE BEEP</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SOUND DOUBLE BEEP</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SOUND_DOUBLE_BEEP
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SOUND_DOUBLE_BEEP_VALUE = 1;

	/**
	 * The '<em><b>SOUND DOWN</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SOUND DOWN</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SOUND_DOWN
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SOUND_DOWN_VALUE = 2;

	/**
	 * The '<em><b>SOUND UP</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SOUND UP</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SOUND_UP
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SOUND_UP_VALUE = 3;

	/**
	 * The '<em><b>SOUND LOW BEEP</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SOUND LOW BEEP</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SOUND_LOW_BEEP
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SOUND_LOW_BEEP_VALUE = 4;

	/**
	 * The '<em><b>SOUND FAST UP</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SOUND FAST UP</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SOUND_FAST_UP
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SOUND_FAST_UP_VALUE = 5;

	/**
	 * An array of all the '<em><b>Sound Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final SoundEnum[] VALUES_ARRAY =
		new SoundEnum[] {
			SOUND_CLICK,
			SOUND_DOUBLE_BEEP,
			SOUND_DOWN,
			SOUND_UP,
			SOUND_LOW_BEEP,
			SOUND_FAST_UP,
		};

	/**
	 * A public read-only list of all the '<em><b>Sound Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<SoundEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Sound Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SoundEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SoundEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Sound Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SoundEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			SoundEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Sound Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static SoundEnum get(int value) {
		switch (value) {
			case SOUND_CLICK_VALUE: return SOUND_CLICK;
			case SOUND_DOUBLE_BEEP_VALUE: return SOUND_DOUBLE_BEEP;
			case SOUND_DOWN_VALUE: return SOUND_DOWN;
			case SOUND_UP_VALUE: return SOUND_UP;
			case SOUND_LOW_BEEP_VALUE: return SOUND_LOW_BEEP;
			case SOUND_FAST_UP_VALUE: return SOUND_FAST_UP;
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
	private SoundEnum(int value, String name, String literal) {
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
	
} //SoundEnum
