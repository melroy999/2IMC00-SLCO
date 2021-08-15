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
 * A representation of the literals of the enumeration '<em><b>Unary Operator Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getUnaryOperatorEnum()
 * @model
 * @generated
 */
public enum UnaryOperatorEnum implements Enumerator {
	/**
	 * The '<em><b>Increase</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #INCREASE_VALUE
	 * @generated
	 * @ordered
	 */
	INCREASE(0, "increase", "increase"),

	/**
	 * The '<em><b>Decrease</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #DECREASE_VALUE
	 * @generated
	 * @ordered
	 */
	DECREASE(1, "decrease", "decrease"),

	/**
	 * The '<em><b>Abs</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ABS_VALUE
	 * @generated
	 * @ordered
	 */
	ABS(2, "abs", "abs"),

	/**
	 * The '<em><b>Sign</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SIGN_VALUE
	 * @generated
	 * @ordered
	 */
	SIGN(3, "sign", "sign"),

	/**
	 * The '<em><b>Umin</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #UMIN_VALUE
	 * @generated
	 * @ordered
	 */
	UMIN(4, "umin", "umin"),

	/**
	 * The '<em><b>Not</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NOT_VALUE
	 * @generated
	 * @ordered
	 */
	NOT(5, "not", "not");

	/**
	 * The '<em><b>Increase</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Increase</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #INCREASE
	 * @model name="increase"
	 * @generated
	 * @ordered
	 */
	public static final int INCREASE_VALUE = 0;

	/**
	 * The '<em><b>Decrease</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Decrease</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #DECREASE
	 * @model name="decrease"
	 * @generated
	 * @ordered
	 */
	public static final int DECREASE_VALUE = 1;

	/**
	 * The '<em><b>Abs</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Abs</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ABS
	 * @model name="abs"
	 * @generated
	 * @ordered
	 */
	public static final int ABS_VALUE = 2;

	/**
	 * The '<em><b>Sign</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Sign</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SIGN
	 * @model name="sign"
	 * @generated
	 * @ordered
	 */
	public static final int SIGN_VALUE = 3;

	/**
	 * The '<em><b>Umin</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Umin</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #UMIN
	 * @model name="umin"
	 * @generated
	 * @ordered
	 */
	public static final int UMIN_VALUE = 4;

	/**
	 * The '<em><b>Not</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Not</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NOT
	 * @model name="not"
	 * @generated
	 * @ordered
	 */
	public static final int NOT_VALUE = 5;

	/**
	 * An array of all the '<em><b>Unary Operator Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final UnaryOperatorEnum[] VALUES_ARRAY =
		new UnaryOperatorEnum[] {
			INCREASE,
			DECREASE,
			ABS,
			SIGN,
			UMIN,
			NOT,
		};

	/**
	 * A public read-only list of all the '<em><b>Unary Operator Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<UnaryOperatorEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Unary Operator Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static UnaryOperatorEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			UnaryOperatorEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Unary Operator Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static UnaryOperatorEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			UnaryOperatorEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Unary Operator Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static UnaryOperatorEnum get(int value) {
		switch (value) {
			case INCREASE_VALUE: return INCREASE;
			case DECREASE_VALUE: return DECREASE;
			case ABS_VALUE: return ABS;
			case SIGN_VALUE: return SIGN;
			case UMIN_VALUE: return UMIN;
			case NOT_VALUE: return NOT;
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
	private UnaryOperatorEnum(int value, String name, String literal) {
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
	
} //UnaryOperatorEnum
