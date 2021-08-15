/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.eclipse.emf.common.util.Enumerator;

/**
 * <!-- begin-user-doc -->
 * A representation of the literals of the enumeration '<em><b>receivetypeenum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see promela.PromelaPackage#getreceivetypeenum()
 * @model
 * @generated
 */
public enum receivetypeenum implements Enumerator {
	/**
	 * The '<em><b>Normal</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NORMAL_VALUE
	 * @generated
	 * @ordered
	 */
	NORMAL(0, "normal", "normal"),

	/**
	 * The '<em><b>Random</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #RANDOM_VALUE
	 * @generated
	 * @ordered
	 */
	RANDOM(1, "random", "random"),

	/**
	 * The '<em><b>Normal poll side effect</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NORMAL_POLL_SIDE_EFFECT_VALUE
	 * @generated
	 * @ordered
	 */
	NORMAL_POLL_SIDE_EFFECT(2, "normal_poll_side_effect", "normal_poll_side_effect"),

	/**
	 * The '<em><b>Random poll side effect</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #RANDOM_POLL_SIDE_EFFECT_VALUE
	 * @generated
	 * @ordered
	 */
	RANDOM_POLL_SIDE_EFFECT(3, "random_poll_side_effect", "random_poll_side_effect");

	/**
	 * The '<em><b>Normal</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Normal</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NORMAL
	 * @model name="normal"
	 * @generated
	 * @ordered
	 */
	public static final int NORMAL_VALUE = 0;

	/**
	 * The '<em><b>Random</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Random</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #RANDOM
	 * @model name="random"
	 * @generated
	 * @ordered
	 */
	public static final int RANDOM_VALUE = 1;

	/**
	 * The '<em><b>Normal poll side effect</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Normal poll side effect</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NORMAL_POLL_SIDE_EFFECT
	 * @model name="normal_poll_side_effect"
	 * @generated
	 * @ordered
	 */
	public static final int NORMAL_POLL_SIDE_EFFECT_VALUE = 2;

	/**
	 * The '<em><b>Random poll side effect</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Random poll side effect</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #RANDOM_POLL_SIDE_EFFECT
	 * @model name="random_poll_side_effect"
	 * @generated
	 * @ordered
	 */
	public static final int RANDOM_POLL_SIDE_EFFECT_VALUE = 3;

	/**
	 * An array of all the '<em><b>receivetypeenum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final receivetypeenum[] VALUES_ARRAY =
		new receivetypeenum[] {
			NORMAL,
			RANDOM,
			NORMAL_POLL_SIDE_EFFECT,
			RANDOM_POLL_SIDE_EFFECT,
		};

	/**
	 * A public read-only list of all the '<em><b>receivetypeenum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<receivetypeenum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>receivetypeenum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static receivetypeenum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			receivetypeenum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>receivetypeenum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static receivetypeenum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			receivetypeenum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>receivetypeenum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static receivetypeenum get(int value) {
		switch (value) {
			case NORMAL_VALUE: return NORMAL;
			case RANDOM_VALUE: return RANDOM;
			case NORMAL_POLL_SIDE_EFFECT_VALUE: return NORMAL_POLL_SIDE_EFFECT;
			case RANDOM_POLL_SIDE_EFFECT_VALUE: return RANDOM_POLL_SIDE_EFFECT;
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
	private receivetypeenum(int value, String name, String literal) {
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
	
} //receivetypeenum
