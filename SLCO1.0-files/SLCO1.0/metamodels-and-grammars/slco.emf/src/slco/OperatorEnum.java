/**
 */
package slco;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.eclipse.emf.common.util.Enumerator;

/**
 * <!-- begin-user-doc -->
 * A representation of the literals of the enumeration '<em><b>Operator Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see slco.SlcoPackage#getOperatorEnum()
 * @model
 * @generated
 */
public enum OperatorEnum implements Enumerator {
	/**
	 * The '<em><b>At Least</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #AT_LEAST_VALUE
	 * @generated
	 * @ordered
	 */
	AT_LEAST(1, "atLeast", "atLeast"),

	/**
	 * The '<em><b>Add</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ADD_VALUE
	 * @generated
	 * @ordered
	 */
	ADD(2, "add", "add"),

	/**
	 * The '<em><b>And</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #AND_VALUE
	 * @generated
	 * @ordered
	 */
	AND(3, "and", "and"),

	/**
	 * The '<em><b>Equals</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EQUALS_VALUE
	 * @generated
	 * @ordered
	 */
	EQUALS(4, "equals", "equals"),

	/**
	 * The '<em><b>Differs</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #DIFFERS_VALUE
	 * @generated
	 * @ordered
	 */
	DIFFERS(5, "differs", "differs"),

	/**
	 * The '<em><b>Subtract</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SUBTRACT_VALUE
	 * @generated
	 * @ordered
	 */
	SUBTRACT(6, "subtract", "subtract"),

	/**
	 * The '<em><b>At Most</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #AT_MOST_VALUE
	 * @generated
	 * @ordered
	 */
	AT_MOST(7, "atMost", "atMost"),

	/**
	 * The '<em><b>Or</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #OR_VALUE
	 * @generated
	 * @ordered
	 */
	OR(8, "or", "or");

	/**
	 * The '<em><b>At Least</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>At Least</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #AT_LEAST
	 * @model name="atLeast"
	 * @generated
	 * @ordered
	 */
	public static final int AT_LEAST_VALUE = 1;

	/**
	 * The '<em><b>Add</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Add</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ADD
	 * @model name="add"
	 * @generated
	 * @ordered
	 */
	public static final int ADD_VALUE = 2;

	/**
	 * The '<em><b>And</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>And</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #AND
	 * @model name="and"
	 * @generated
	 * @ordered
	 */
	public static final int AND_VALUE = 3;

	/**
	 * The '<em><b>Equals</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Equals</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EQUALS
	 * @model name="equals"
	 * @generated
	 * @ordered
	 */
	public static final int EQUALS_VALUE = 4;

	/**
	 * The '<em><b>Differs</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Differs</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #DIFFERS
	 * @model name="differs"
	 * @generated
	 * @ordered
	 */
	public static final int DIFFERS_VALUE = 5;

	/**
	 * The '<em><b>Subtract</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Subtract</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SUBTRACT
	 * @model name="subtract"
	 * @generated
	 * @ordered
	 */
	public static final int SUBTRACT_VALUE = 6;

	/**
	 * The '<em><b>At Most</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>At Most</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #AT_MOST
	 * @model name="atMost"
	 * @generated
	 * @ordered
	 */
	public static final int AT_MOST_VALUE = 7;

	/**
	 * The '<em><b>Or</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Or</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #OR
	 * @model name="or"
	 * @generated
	 * @ordered
	 */
	public static final int OR_VALUE = 8;

	/**
	 * An array of all the '<em><b>Operator Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final OperatorEnum[] VALUES_ARRAY =
		new OperatorEnum[] {
			AT_LEAST,
			ADD,
			AND,
			EQUALS,
			DIFFERS,
			SUBTRACT,
			AT_MOST,
			OR,
		};

	/**
	 * A public read-only list of all the '<em><b>Operator Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<OperatorEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Operator Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static OperatorEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			OperatorEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Operator Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static OperatorEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			OperatorEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Operator Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static OperatorEnum get(int value) {
		switch (value) {
			case AT_LEAST_VALUE: return AT_LEAST;
			case ADD_VALUE: return ADD;
			case AND_VALUE: return AND;
			case EQUALS_VALUE: return EQUALS;
			case DIFFERS_VALUE: return DIFFERS;
			case SUBTRACT_VALUE: return SUBTRACT;
			case AT_MOST_VALUE: return AT_MOST;
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
	private OperatorEnum(int value, String name, String literal) {
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
	
} //OperatorEnum
