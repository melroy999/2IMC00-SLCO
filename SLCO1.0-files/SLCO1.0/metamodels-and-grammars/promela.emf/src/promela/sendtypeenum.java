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
 * A representation of the literals of the enumeration '<em><b>sendtypeenum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see promela.PromelaPackage#getsendtypeenum()
 * @model
 * @generated
 */
public enum sendtypeenum implements Enumerator {
	/**
	 * The '<em><b>Normal fifo</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NORMAL_FIFO_VALUE
	 * @generated
	 * @ordered
	 */
	NORMAL_FIFO(0, "normal_fifo", "normal_fifo"),

	/**
	 * The '<em><b>Sorted</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SORTED_VALUE
	 * @generated
	 * @ordered
	 */
	SORTED(1, "sorted", "sorted");

	/**
	 * The '<em><b>Normal fifo</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Normal fifo</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NORMAL_FIFO
	 * @model name="normal_fifo"
	 * @generated
	 * @ordered
	 */
	public static final int NORMAL_FIFO_VALUE = 0;

	/**
	 * The '<em><b>Sorted</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Sorted</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SORTED
	 * @model name="sorted"
	 * @generated
	 * @ordered
	 */
	public static final int SORTED_VALUE = 1;

	/**
	 * An array of all the '<em><b>sendtypeenum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final sendtypeenum[] VALUES_ARRAY =
		new sendtypeenum[] {
			NORMAL_FIFO,
			SORTED,
		};

	/**
	 * A public read-only list of all the '<em><b>sendtypeenum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<sendtypeenum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>sendtypeenum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static sendtypeenum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			sendtypeenum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>sendtypeenum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static sendtypeenum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			sendtypeenum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>sendtypeenum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static sendtypeenum get(int value) {
		switch (value) {
			case NORMAL_FIFO_VALUE: return NORMAL_FIFO;
			case SORTED_VALUE: return SORTED;
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
	private sendtypeenum(int value, String name, String literal) {
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
	
} //sendtypeenum
