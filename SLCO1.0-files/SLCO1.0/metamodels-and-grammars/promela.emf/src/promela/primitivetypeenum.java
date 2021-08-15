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
 * A representation of the literals of the enumeration '<em><b>primitivetypeenum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see promela.PromelaPackage#getprimitivetypeenum()
 * @model
 * @generated
 */
public enum primitivetypeenum implements Enumerator {
	/**
	 * The '<em><b>BIT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #BIT_VALUE
	 * @generated
	 * @ordered
	 */
	BIT(0, "BIT", "BIT"),

	/**
	 * The '<em><b>BOOL</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #BOOL_VALUE
	 * @generated
	 * @ordered
	 */
	BOOL(1, "BOOL", "BOOL"),

	/**
	 * The '<em><b>BYTE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #BYTE_VALUE
	 * @generated
	 * @ordered
	 */
	BYTE(2, "BYTE", "BYTE"),

	/**
	 * The '<em><b>SHORT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SHORT_VALUE
	 * @generated
	 * @ordered
	 */
	SHORT(3, "SHORT", "SHORT"),

	/**
	 * The '<em><b>INT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #INT_VALUE
	 * @generated
	 * @ordered
	 */
	INT(4, "INT", "INT"),

	/**
	 * The '<em><b>MTYPE</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #MTYPE_VALUE
	 * @generated
	 * @ordered
	 */
	MTYPE(5, "MTYPE", "MTYPE"),

	/**
	 * The '<em><b>CHAN</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #CHAN_VALUE
	 * @generated
	 * @ordered
	 */
	CHAN(6, "CHAN", "CHAN");

	/**
	 * The '<em><b>BIT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>BIT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #BIT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int BIT_VALUE = 0;

	/**
	 * The '<em><b>BOOL</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>BOOL</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #BOOL
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int BOOL_VALUE = 1;

	/**
	 * The '<em><b>BYTE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>BYTE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #BYTE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int BYTE_VALUE = 2;

	/**
	 * The '<em><b>SHORT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>SHORT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SHORT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int SHORT_VALUE = 3;

	/**
	 * The '<em><b>INT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>INT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #INT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int INT_VALUE = 4;

	/**
	 * The '<em><b>MTYPE</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>MTYPE</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #MTYPE
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int MTYPE_VALUE = 5;

	/**
	 * The '<em><b>CHAN</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>CHAN</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #CHAN
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int CHAN_VALUE = 6;

	/**
	 * An array of all the '<em><b>primitivetypeenum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final primitivetypeenum[] VALUES_ARRAY =
		new primitivetypeenum[] {
			BIT,
			BOOL,
			BYTE,
			SHORT,
			INT,
			MTYPE,
			CHAN,
		};

	/**
	 * A public read-only list of all the '<em><b>primitivetypeenum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<primitivetypeenum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>primitivetypeenum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static primitivetypeenum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			primitivetypeenum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>primitivetypeenum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static primitivetypeenum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			primitivetypeenum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>primitivetypeenum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static primitivetypeenum get(int value) {
		switch (value) {
			case BIT_VALUE: return BIT;
			case BOOL_VALUE: return BOOL;
			case BYTE_VALUE: return BYTE;
			case SHORT_VALUE: return SHORT;
			case INT_VALUE: return INT;
			case MTYPE_VALUE: return MTYPE;
			case CHAN_VALUE: return CHAN;
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
	private primitivetypeenum(int value, String name, String literal) {
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
	
} //primitivetypeenum
