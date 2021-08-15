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
 * A representation of the literals of the enumeration '<em><b>binarop</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see promela.PromelaPackage#getbinarop()
 * @model
 * @generated
 */
public enum binarop implements Enumerator {
	/**
	 * The '<em><b>PLUS</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #PLUS_VALUE
	 * @generated
	 * @ordered
	 */
	PLUS(0, "PLUS", "PLUS"),

	/**
	 * The '<em><b>MINUS</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #MINUS_VALUE
	 * @generated
	 * @ordered
	 */
	MINUS(1, "MINUS", "MINUS"),

	/**
	 * The '<em><b>TIMES</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #TIMES_VALUE
	 * @generated
	 * @ordered
	 */
	TIMES(2, "TIMES", "TIMES"),

	/**
	 * The '<em><b>DIV</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #DIV_VALUE
	 * @generated
	 * @ordered
	 */
	DIV(3, "DIV", "DIV"),

	/**
	 * The '<em><b>MOD</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #MOD_VALUE
	 * @generated
	 * @ordered
	 */
	MOD(4, "MOD", "MOD"),

	/**
	 * The '<em><b>AMPERSAND</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #AMPERSAND_VALUE
	 * @generated
	 * @ordered
	 */
	AMPERSAND(5, "AMPERSAND", "AMPERSAND"),

	/**
	 * The '<em><b>CARET</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CARET_VALUE
	 * @generated
	 * @ordered
	 */
	CARET(6, "CARET", "CARET"),

	/**
	 * The '<em><b>BAR</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #BAR_VALUE
	 * @generated
	 * @ordered
	 */
	BAR(7, "BAR", "BAR"),

	/**
	 * The '<em><b>GREATER</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #GREATER_VALUE
	 * @generated
	 * @ordered
	 */
	GREATER(8, "GREATER", "GREATER"),

	/**
	 * The '<em><b>SMALLER</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SMALLER_VALUE
	 * @generated
	 * @ordered
	 */
	SMALLER(9, "SMALLER", "SMALLER"),

	/**
	 * The '<em><b>GEQ</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #GEQ_VALUE
	 * @generated
	 * @ordered
	 */
	GEQ(10, "GEQ", "GEQ"),

	/**
	 * The '<em><b>LEQ</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #LEQ_VALUE
	 * @generated
	 * @ordered
	 */
	LEQ(11, "LEQ", "LEQ"),

	/**
	 * The '<em><b>EQUALS</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EQUALS_VALUE
	 * @generated
	 * @ordered
	 */
	EQUALS(12, "EQUALS", "EQUALS"),

	/**
	 * The '<em><b>DIFFERS</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #DIFFERS_VALUE
	 * @generated
	 * @ordered
	 */
	DIFFERS(13, "DIFFERS", "DIFFERS"),

	/**
	 * The '<em><b>SHIFT RIGHT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SHIFT_RIGHT_VALUE
	 * @generated
	 * @ordered
	 */
	SHIFT_RIGHT(14, "SHIFT_RIGHT", "SHIFT_RIGHT"),

	/**
	 * The '<em><b>SHIFT LEFT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SHIFT_LEFT_VALUE
	 * @generated
	 * @ordered
	 */
	SHIFT_LEFT(15, "SHIFT_LEFT", "SHIFT_LEFT"),

	/**
	 * The '<em><b>AND</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #AND_VALUE
	 * @generated
	 * @ordered
	 */
	AND(16, "AND", "AND"),

	/**
	 * The '<em><b>OR</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OR_VALUE
	 * @generated
	 * @ordered
	 */
	OR(17, "OR", "OR");

	/**
	 * The '<em><b>PLUS</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>PLUS</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #PLUS
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int PLUS_VALUE = 0;

	/**
	 * The '<em><b>MINUS</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>MINUS</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #MINUS
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int MINUS_VALUE = 1;

	/**
	 * The '<em><b>TIMES</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>TIMES</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #TIMES
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int TIMES_VALUE = 2;

	/**
	 * The '<em><b>DIV</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>DIV</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #DIV
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int DIV_VALUE = 3;

	/**
	 * The '<em><b>MOD</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>MOD</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #MOD
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int MOD_VALUE = 4;

	/**
	 * The '<em><b>AMPERSAND</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>AMPERSAND</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #AMPERSAND
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int AMPERSAND_VALUE = 5;

	/**
	 * The '<em><b>CARET</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>CARET</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CARET
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int CARET_VALUE = 6;

	/**
	 * The '<em><b>BAR</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>BAR</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #BAR
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int BAR_VALUE = 7;

	/**
	 * The '<em><b>GREATER</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>GREATER</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #GREATER
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int GREATER_VALUE = 8;

	/**
	 * The '<em><b>SMALLER</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SMALLER</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SMALLER
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SMALLER_VALUE = 9;

	/**
	 * The '<em><b>GEQ</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>GEQ</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #GEQ
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int GEQ_VALUE = 10;

	/**
	 * The '<em><b>LEQ</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>LEQ</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #LEQ
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int LEQ_VALUE = 11;

	/**
	 * The '<em><b>EQUALS</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EQUALS</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EQUALS
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EQUALS_VALUE = 12;

	/**
	 * The '<em><b>DIFFERS</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>DIFFERS</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #DIFFERS
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int DIFFERS_VALUE = 13;

	/**
	 * The '<em><b>SHIFT RIGHT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SHIFT RIGHT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SHIFT_RIGHT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SHIFT_RIGHT_VALUE = 14;

	/**
	 * The '<em><b>SHIFT LEFT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SHIFT LEFT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SHIFT_LEFT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SHIFT_LEFT_VALUE = 15;

	/**
	 * The '<em><b>AND</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>AND</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #AND
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int AND_VALUE = 16;

	/**
	 * The '<em><b>OR</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>OR</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OR
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int OR_VALUE = 17;

	/**
	 * An array of all the '<em><b>binarop</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final binarop[] VALUES_ARRAY =
		new binarop[] {
			PLUS,
			MINUS,
			TIMES,
			DIV,
			MOD,
			AMPERSAND,
			CARET,
			BAR,
			GREATER,
			SMALLER,
			GEQ,
			LEQ,
			EQUALS,
			DIFFERS,
			SHIFT_RIGHT,
			SHIFT_LEFT,
			AND,
			OR,
		};

	/**
	 * A public read-only list of all the '<em><b>binarop</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<binarop> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>binarop</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static binarop get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			binarop result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>binarop</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static binarop getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			binarop result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>binarop</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static binarop get(int value) {
		switch (value) {
			case PLUS_VALUE: return PLUS;
			case MINUS_VALUE: return MINUS;
			case TIMES_VALUE: return TIMES;
			case DIV_VALUE: return DIV;
			case MOD_VALUE: return MOD;
			case AMPERSAND_VALUE: return AMPERSAND;
			case CARET_VALUE: return CARET;
			case BAR_VALUE: return BAR;
			case GREATER_VALUE: return GREATER;
			case SMALLER_VALUE: return SMALLER;
			case GEQ_VALUE: return GEQ;
			case LEQ_VALUE: return LEQ;
			case EQUALS_VALUE: return EQUALS;
			case DIFFERS_VALUE: return DIFFERS;
			case SHIFT_RIGHT_VALUE: return SHIFT_RIGHT;
			case SHIFT_LEFT_VALUE: return SHIFT_LEFT;
			case AND_VALUE: return AND;
			case OR_VALUE: return OR;
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
	private binarop(int value, String name, String literal) {
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
	
} //binarop
