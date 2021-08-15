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
 * A representation of the literals of the enumeration '<em><b>polltypeenum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see promela.PromelaPackage#getpolltypeenum()
 * @model
 * @generated
 */
public enum polltypeenum implements Enumerator {
	/**
	 * The '<em><b>Normal poll no side effect</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NORMAL_POLL_NO_SIDE_EFFECT_VALUE
	 * @generated
	 * @ordered
	 */
	NORMAL_POLL_NO_SIDE_EFFECT(0, "normal_poll_no_side_effect", "normal_poll_no_side_effect"),

	/**
	 * The '<em><b>Random poll no side effect</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #RANDOM_POLL_NO_SIDE_EFFECT_VALUE
	 * @generated
	 * @ordered
	 */
	RANDOM_POLL_NO_SIDE_EFFECT(1, "random_poll_no_side_effect", "random_poll_no_side_effect");

	/**
	 * The '<em><b>Normal poll no side effect</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Normal poll no side effect</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NORMAL_POLL_NO_SIDE_EFFECT
	 * @model name="normal_poll_no_side_effect"
	 * @generated
	 * @ordered
	 */
	public static final int NORMAL_POLL_NO_SIDE_EFFECT_VALUE = 0;

	/**
	 * The '<em><b>Random poll no side effect</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Random poll no side effect</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #RANDOM_POLL_NO_SIDE_EFFECT
	 * @model name="random_poll_no_side_effect"
	 * @generated
	 * @ordered
	 */
	public static final int RANDOM_POLL_NO_SIDE_EFFECT_VALUE = 1;

	/**
	 * An array of all the '<em><b>polltypeenum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final polltypeenum[] VALUES_ARRAY =
		new polltypeenum[] {
			NORMAL_POLL_NO_SIDE_EFFECT,
			RANDOM_POLL_NO_SIDE_EFFECT,
		};

	/**
	 * A public read-only list of all the '<em><b>polltypeenum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<polltypeenum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>polltypeenum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static polltypeenum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			polltypeenum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>polltypeenum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static polltypeenum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			polltypeenum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>polltypeenum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static polltypeenum get(int value) {
		switch (value) {
			case NORMAL_POLL_NO_SIDE_EFFECT_VALUE: return NORMAL_POLL_NO_SIDE_EFFECT;
			case RANDOM_POLL_NO_SIDE_EFFECT_VALUE: return RANDOM_POLL_NO_SIDE_EFFECT;
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
	private polltypeenum(int value, String name, String literal) {
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
	
} //polltypeenum
